import ast
import json
import logging
import math
import os
import re
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
import pickle
import csv


try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from dataloaders import cifar10, cifar100, dtd, food101, stanford_car, fgvc_aircraft, flowers102, oxford_pets, caltech101, sun397

module_dict = {
    "food101": food101,  
    "cifar10": cifar10,
    "cifar100": cifar100,
    "sun397": sun397,
    "stanford_car": stanford_car,
    "aircraft": fgvc_aircraft,
    "dtd": dtd,
    "pets": oxford_pets,
    "flowers": flowers102,
    "caltech101": caltech101,
}


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist), \
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)])
                         for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(
            location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = (
        'png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def filter_no_caption_or_no_image_json(sample):
    has_caption = ('json' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def split_caption(text):
    """Split captions by sentence-ending markers."""
    return [cap.strip() for cap in re.split(r'\n|</s>|[.]', text) if cap.strip()]


def sample_dict(text, k=3, tokenizer=None, sampling_mode='random'):
    '''
    :param text: Should be dict, each containing a list
    :param k:
    :param tokenizer:
    :param sampling_mode:
    :return:
    '''
    # raw_caption, shortIB_captions, longIB_captions, shortSV_captions, longSV_captions, shortLLA_captions, longLLA_captions
    sampled_sentences = None

    if sampling_mode == 'raw':
        # 适配我们的简单文本格式，允许多次采样
        if isinstance(text, str):
            # 对于简单字符串，重复k次以支持多采样
            captions_list = [text] * k
        elif 'raw_caption' in text:
            captions_list = text['raw_caption']
            if len(captions_list) < k:
                # 如果原始数据不够，重复填充
                captions_list = captions_list * ((k // len(captions_list)) + 1)
        else:
            # 假设text是我们的简单字符串，重复k次
            caption = text if isinstance(text, str) else str(text)
            captions_list = [caption] * k
    elif sampling_mode == 'raw_pixelprose':
        sampled_sentences = [text['original_caption']]
    elif sampling_mode == 'random':
        captions_list = text['raw_caption'] + text['shortIB_captions'] + text['shortSV_captions'] + text['shortLLA_captions'] + \
                               text['longIB_captions'] + text['longSV_captions'] + text['longLLA_captions']
    elif sampling_mode == 'random_pixelprose':
        captions_list = [text['original_caption']] + split_caption(text=text['caption'])
    elif sampling_mode == 'short':
        captions_list = text['raw_caption'] + text['shortIB_captions'] + text['shortSV_captions'] + text['shortLLA_captions']
    elif sampling_mode == 'long':
        captions_list = text['longIB_captions'] + text['longSV_captions'] + text['longLLA_captions']
    elif sampling_mode == 'textcrop':
        assert k >= 2
        captions_list = text['raw_caption'] + \
                            text['shortIB_captions'] + text['shortSV_captions'] + text['shortLLA_captions'] + \
                            text['longIB_captions'] + text['longSV_captions'] + text['longLLA_captions']
        global_nums = [random.randint(1, 5) for _ in range(2) ] # choose the number of sentences (1~5) for 2 global captions
        global_captions = ['. '.join(random_sample_from_list(captions_list, num)) for num in global_nums]
        local_captions = random_sample_from_list(captions_list, k-2)
        sampled_sentences = global_captions + local_captions
    elif sampling_mode == 'textcrop_pixelprose':
        assert k >= 2
        captions_list = [text['original_caption']] + split_caption(text=text['caption'])
        global_nums = [random.randint(1,5) for _ in range(2) ] # choose the number of sentences (1~5) for 2 global captions
        global_captions = ['. '.join(random_sample_from_list(captions_list, num)) for num in global_nums]
        local_captions = random_sample_from_list(captions_list, k-2)
        sampled_sentences = global_captions + local_captions
    elif sampling_mode == 'multi_captions':
        # For datasets with multiple captions in 'original_captions' array
        assert k >= 2
        if 'original_captions' in text:
            captions_list = text['original_captions']
        else:
            # Fallback to combined caption
            captions_list = split_caption(text=text.get('combined_caption', text.get('caption', '')))
        
        # Sample k captions (with repetition if needed)
        sampled_sentences = random_sample_from_list(captions_list, k)
    else:
        raise NotImplementedError('Please select a valid sampling method')
    
    if sampled_sentences is None:
        sampled_sentences = random_sample_from_list(captions_list, k)
    tokenized_sentences = tokenizer(sampled_sentences) # currently assume we are using SimpleTokenizer()
    return tokenized_sentences


def random_sample_from_list(captions_list, num_cap):
    n = len(captions_list)
    if n >= num_cap:
        return random.sample(captions_list, num_cap)
    else: # minimizing caption dupilications
        div = num_cap // n
        remain = num_cap % n
        return div * captions_list + random.sample(captions_list, remain)


def mask_words(text, prob):
    words = text.split()
    masked_words = [word if random.random() > prob else 'mask' for word in words]
    return ' '.join(masked_words)


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(
                __key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights), \
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None, train_eval=False):
    if is_train:
        input_shards = args.train_data
    else:
        if train_eval:
            input_shards = args.train_eval_data
        else:
            input_shards = args.val_data

    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        if train_eval:
            num_samples = args.train_val_num_samples or 0
        else:
            num_samples = args.val_num_samples or 0

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."

    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            wds.tarfile_to_samples(handler=log_and_continue),
            # tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])

    if args.num_sampled_captions:
        if is_train:
            pipeline.extend([
                wds.select(filter_no_caption_or_no_image_json),
                wds.decode("pilrgb", handler=log_and_continue),
                wds.rename(image="jpg;png;jpeg;webp", text="json"),
                wds.map_dict(image=preprocess_img, text=lambda text: sample_dict(text, k=args.num_sampled_captions, tokenizer=tokenizer, sampling_mode=args.caption_sampling_mode)),
                wds.to_tuple("image", "text"),
                wds.batched(args.batch_size, partial=not is_train)
            ])
        else:
            pipeline.extend([
                wds.select(filter_no_caption_or_no_image),
                wds.decode("pilrgb", handler=log_and_continue),
                wds.rename(image="jpg;png;jpeg;webp", text="txt"),
                wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
                wds.to_tuple("image", "text"),
                wds.batched(args.batch_size, partial=not is_train)
            ])
            
    else:
        pipeline.extend([
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
            wds.to_tuple("image", "text"),
            wds.batched(args.batch_size, partial=not is_train)
        ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        # each worker is iterating over this
        dataset = dataset.with_epoch(num_worker_batches)
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(
        dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_coco_train_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    data_root_dir = args.train_data
    split = 'train'

    dataset = COCOCaptionDataset(
        root_dir=data_root_dir, transform=preprocess_fn, split=split, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(
        dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_coco_dataset(args, preprocess_fn, tokenizer=None, root_dir=None):
    data_root_dir = args.data_root_dir if root_dir is None else root_dir

    split = 'val'
    sampler = None
    shuffle = False

    txt_dataset = COCOTextDataset(root_dir=data_root_dir, transform=preprocess_fn,
                                  split=split, tokenizer=tokenizer, sampling_mode=None, num_samples=None)
    img2txt_dict, txt2img_dict = txt_dataset.img2txt_dict, txt_dataset.txt2img_dict
    num_txt_samples = len(txt_dataset)

    img_dataset = COCOImageDataset(root_dir=data_root_dir, data_list=txt_dataset.data_list, transform=preprocess_fn,
                                   split=split)
    num_img_samples = len(img_dataset)

    # drop_last = is_train or args.text_conditioned_loss
    # if we used text_conditioned_loss, then we always drop_last

    txt_dataloader = DataLoader(
        txt_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    #TODO: This time we cannot use 'drop_last'

    txt_dataloader.num_samples = num_txt_samples
    txt_dataloader.num_batches = len(txt_dataloader)

    img_dataloader = DataLoader(
        img_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    img_dataloader.num_samples = num_img_samples
    img_dataloader.num_batches = len(img_dataloader)

    return DataInfo(txt_dataloader, sampler), DataInfo(img_dataloader, sampler), img2txt_dict, txt2img_dict


class COCOTextDataset(Dataset):
    '''
    Only loading captions and captions ID. Used in Text-conditioned setting. Only in validation
    '''

    def __init__(self, root_dir, dict_root_dir=None, transform=None, split='train', tokenizer=None, sampling_mode=None,
                 num_samples=None):
        self.root_dir = root_dir
        self.dict_root_dir = dict_root_dir
        self.transform = transform
        self.split = split
        self.sampling_mode = sampling_mode
        self.num_samples = num_samples
        # create data list
        logging.info(f"creating dataset list...")
        data_list = read_coco_pairs(root_dir=self.root_dir, split=self.split)
        logging.info(f"dataset list created, pretokenizing...")
        self.data_list = pre_tokenize(tokenizer=tokenizer, data_list=data_list)
        logging.info(f"pretokenization finished...")
        if self.split == 'val':
            self.img2txt_dict, self.txt2img_dict = map_img_cap(self.data_list)
            logging.info(f"In validation mode, finish constructing the img_cap mapping dict for retrieval")
            # adding two dictionaries indicating the mapping between image index and text index

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        img_id = data["image_id"]

        caption = data["caption"].squeeze(dim=0)

        if self.split == 'val':
            cap_id = data["caption_id"]
            return caption, cap_id  # Only retruning captions and cap_ids
        else:
            return caption


class COCOImageDataset(Dataset):
    '''
    Only loading images and img_ids. Used in Text-conditioned setting. Only in validation
    '''

    def __init__(self, root_dir, data_list=None, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        # create data list
        logging.info(f"reusing pre-tokenized datalist that we get from COCOTextDataset, extracting...")
        self.img_list = extract_unique_img_list_from_data_list(data_list=data_list)
        logging.info(f"finish extracting all unique images with img_ids from the whole data_list")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        data = self.img_list[idx]
        img_id = data["image_id"]
        img_path = data["image"]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_id


def get_flickr_train_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    data_root_dir = args.train_data
    if is_train:
        split = 'train'
    else:
        split = 'test'
    logging.info(f"loading {split} set for flickr")

    dataset = FlickrCaptionDataset(root_dir=data_root_dir, transform=preprocess_fn, split=split, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_flickr_dataset(args, preprocess_fn, tokenizer=None, root_dir=None):
    data_root_dir = args.data_root_dir if root_dir is None else root_dir
    # if we provide the 'dict_root_dir' in args, meaning that we want to filter the dataset
    # as default, the 'dict_root_dir' in params should be None
    split = 'val'
    sampler = None
    shuffle = False

    txt_dataset = FlickrTextDataset(root_dir=data_root_dir, transform=preprocess_fn,
                                    split=split, tokenizer=tokenizer)
    img2txt_dict, txt2img_dict = txt_dataset.img2txt_dict, txt_dataset.txt2img_dict
    num_txt_samples = len(txt_dataset)

    img_dataset = FlickrImageDataset(root_dir=data_root_dir, data_list=txt_dataset.data_list, transform=preprocess_fn,
                                     split=split)
    num_img_samples = len(img_dataset)

    # drop_last = is_train or args.text_conditioned_loss
    # if we used text_conditioned_loss, then we always drop_last

    txt_dataloader = DataLoader(
        txt_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    # TODO: This time we cannot use 'drop_last'

    txt_dataloader.num_samples = num_txt_samples
    txt_dataloader.num_batches = len(txt_dataloader)

    img_dataloader = DataLoader(
        img_dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    img_dataloader.num_samples = num_img_samples
    img_dataloader.num_batches = len(img_dataloader)

    return DataInfo(txt_dataloader, sampler), DataInfo(img_dataloader, sampler), img2txt_dict, txt2img_dict


class FlickrTextDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', tokenizer=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        # create data list
        logging.info(f"creating dataset list...")
        data_list = read_flickr_pairs(root_dir=self.root_dir, split=self.split)
        #logging.info(f"dataset list created, pretokenizing...")
        self.data_list = pre_tokenize(tokenizer=tokenizer, data_list=data_list)
        logging.info(f"pretokenization finished...")
        if self.split == 'val':
            self.img2txt_dict, self.txt2img_dict = map_img_cap(self.data_list)
            logging.info(f"In validation mode, finish constructing the img_cap mapping dict for retrieval")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        caption = data["caption"].squeeze(dim=0)
        if self.split == 'val':
            cap_id = data["caption_id"]
            return caption, cap_id
        else:
            return caption


class FlickrImageDataset(Dataset):
    '''
    Only loading images and img_ids. Used in Text-conditioned setting. Only in validation
    '''

    def __init__(self, root_dir, data_list=None, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        # create data list
        logging.info(f"reusing pre-tokenized datalist that we get from FlickrTextDataset, extracting...")
        self.img_list = extract_unique_img_list_from_data_list(data_list=data_list)
        logging.info(f"finish extracting all unique images with img_ids from the whole data_list")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        data = self.img_list[idx]
        img_id = data["image_id"]
        img_path = data["image"]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_id


def extract_unique_img_list_from_data_list(data_list):
    """
    :param data_list: a list of dicts, each: {'image', 'image_id', 'caption', 'caption_id'}
    :return: img_list: a list of dicts, with all unique 'image' w.r.t. 'image_id'. So each new dict will be {'image', 'image_id'}
    """
    seen_ids = set()
    img_list = []

    for item in data_list:
        image_id = item['image_id']
        if image_id not in seen_ids:
            # Add to the list and mark the id as seen
            img_list.append({'image': item['image'], 'image_id': image_id})
            seen_ids.add(image_id)

    return img_list


def get_dataset_fn(dataset_type):
    if dataset_type == "coco":
        return get_coco_train_dataset
    elif dataset_type == "flickr":
        return get_flickr_train_dataset
    elif dataset_type == "webdataset":
        return get_wds_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_dataset_fn(args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)
        
    if args.train_eval_data:
        data["train_eval"] = get_dataset_fn(args.dataset_type)(
            args, preprocess_val, is_train=False, epoch=epoch, tokenizer=tokenizer, train_eval=True)

    if args.val_data == 'retrieval':
        data["val_coco"] = get_coco_dataset(
            args, preprocess_val, tokenizer=tokenizer, root_dir=os.path.join(args.data_root_dir, 'coco'))
        data["val_flickr"] = get_flickr_dataset(
            args, preprocess_val, tokenizer=tokenizer, root_dir=os.path.join(args.data_root_dir, 'flickr30k-images'))
    elif args.val_data == 'classification':
        # data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")
        dataset_names = ["sun397"] # ["food101", "cifar10", "cifar100", "sun397", "stanford_car", "aircraft", "dtd", "pets", "caltech101", "flowers"]
        for name in dataset_names:
            dataset_module = module_dict[name]
            
            dataset = dataset_module.get_loader_test(preprocess_val, batch_size=None, num_workers=None, seed=3072)[0] 
            data[name] = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.workers,
            )
    elif args.val_data == 'coco':
        data["val_coco"] = get_coco_dataset(
            args, preprocess_val, tokenizer=tokenizer, root_dir=os.path.join(args.data_root_dir, 'coco')) 
    elif args.val_data == 'flickr':
        data["val_flickr"] = get_flickr_dataset(
            args, preprocess_val, tokenizer=tokenizer, root_dir=os.path.join(args.data_root_dir, 'flickr30k-images'))
    elif args.val_data:
        data["val"] = get_dataset_fn(args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    return data


def pre_tokenize(tokenizer, data_list):
    for data in data_list:
        data["caption"] = tokenizer(data["caption"])
    return data_list


def read_coco_pairs(root_dir, split='train'):
    """
    :param num_samples: int
    :param sampling_mode: str.
    :param root_dir: str; path to the dataset folder
    :param dict_root_dir: str; path to the preprocessed dictionaries
    :param split: str; 'train' or 'val'
    :return: a list of dict: {'image_id': int, 'image': str, 'caption': str}
    """
    annotations_dir = os.path.join(root_dir, "annotations")
    if split == "train":
        captions_file = os.path.join(annotations_dir, "captions_train2017.json")
        images_dir = os.path.join(root_dir, "images", "train2017")
    else:
        split = 'val'
        captions_file = os.path.join(annotations_dir, "captions_val2017.json")
        images_dir = os.path.join(root_dir, "images", "val2017")

    with open(captions_file, 'r') as f:
        coco_data = json.load(f)

    image_id_to_path = {image['id']: os.path.join(images_dir, image['file_name']) for image in coco_data['images']}
    # the case for validation set
    data_list = []
    cap_id = 0
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id in image_id_to_path:
            data_list.append({
                'image_id': image_id,
                'image': image_id_to_path[image_id],
                'caption': annotation['caption'],
                'caption_id': cap_id
            })
        cap_id += 1
    # data_list = sorted(data_list, key=lambda x: x['image_id'])

    return data_list


def map_img_cap(data_list):
    """
    :param data_list: List of dict, each dict contains key 'image_id' and 'caption_id'
    :return: img2txt_dict, txt2img_dict
    """
    img2txt_dict = {}
    txt2img_dict = {}

    for entry in data_list:
        image_id = entry['image_id']
        caption_id = entry['caption_id']

        if image_id not in img2txt_dict:
            img2txt_dict[image_id] = [caption_id]
        else:
            img2txt_dict[image_id].append(caption_id)

        if caption_id not in txt2img_dict:
            txt2img_dict[caption_id] = [image_id]
        else:
            txt2img_dict[caption_id].append(image_id)
    return img2txt_dict, txt2img_dict


class COCOCaptionDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', tokenizer=None):
        self.root_dir = root_dir
        self.transform = transform
        if split == 'train_eval':
            self.split = 'train'
        else:
            self.split = split

        # create data list
        logging.info("creating dataset list...")
        data_list = read_coco_pairs(root_dir=self.root_dir, split=self.split)
        if split == 'train_eval':
            data_list = data_list[:5120]
        logging.info("dataset list created, pretokenizing...")
        self.data_list = pre_tokenize(tokenizer=tokenizer, data_list=data_list)
        logging.info("pretokenization finished...")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        img_id = data["image_id"]
        img_path = data["image"]

        caption = data["caption"].squeeze(dim=0)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, caption


def read_flickr_pairs(root_dir, split='train'):
    base_dir = os.path.dirname(root_dir)
    if split == 'train':
        captions_file = os.path.join(root_dir, "flickr30k_train.json")
    elif split == 'val':
        captions_file = os.path.join(root_dir, "flickr30k_val.json")
    else:
        captions_file = os.path.join(root_dir, "flickr30k_test.json")

    with open(captions_file, 'r') as f:
        flickr_data = json.load(f)

    data_list = []
    img_id, cap_id = 0, 0
    for annotation in flickr_data:
        image_path = os.path.join(base_dir, annotation['image'])
        caption_list = annotation["caption"]  # Now the caption should be a list
        if isinstance(caption_list, list):
            for caption in caption_list:
                data_list.append({
                    'image': image_path,
                    'caption': caption,
                    'image_id': img_id,
                    'caption_id': cap_id
                })
                cap_id += 1
            img_id += 1
        else:
            data_list.append({
                'image': image_path,
                'caption': caption_list
            })            
    return data_list


class FlickrCaptionDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', tokenizer=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        # create data list
        logging.info("creating dataset list...")
        data_list = read_flickr_pairs(root_dir=self.root_dir, split=self.split)
        logging.info("dataset list created, pretokenizing...")
        self.data_list = pre_tokenize(tokenizer=tokenizer, data_list=data_list)
        logging.info("pretokenization finished...")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        img_path = data["image"]
        caption = data["caption"].squeeze(dim=0)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, caption
