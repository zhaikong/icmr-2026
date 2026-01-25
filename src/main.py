import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from huggingface_hub import hf_hub_download

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from training.data import get_data
from training.distributed import is_master, init_distributed_device, broadcast_object
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown, cosine_scheduler
from training.train import train_one_epoch, evaluate, zeroshot_evaluate_retrieval, zeroshot_evaluate_classification
from training.file_utils import pt_load, check_exists, start_sync_process, remote_sync

import warnings

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(
            ["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1])
                       for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None

def download_weights_from_hf(model_repo, filename):
    # Define the custom cache directory relative to the current script
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    local_path = hf_hub_download(repo_id=model_repo, filename=filename, cache_dir=cache_dir)
    return local_path

def main(args):
    args = parse_args(args)

    # Increase the maximum number of file descriptors to resolve
    # RuntimeError: received 0 items of ancdata
    # https://stackoverflow.com/questions/71642653/how-to-resolve-the-error-runtimeerror-received-0-items-of-ancdata
    warnings.filterwarnings("ignore", message="Corrupt EXIF data")
    warnings.filterwarnings("ignore", message="Truncated File Read")

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
            f"key_{args.wandbkeyword}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs_dir, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(
            log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(
                args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print(
                    'Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(
                    checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(
                    checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(
                    f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(
                    f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs_dir, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs_dir, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        # FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        # FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model_kwargs = {}

    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10
    student, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        use_imagecrop_aug=args.use_imagecrop_aug,
        global_crops_number=args.global_crops_number,
        local_crops_number=args.local_crops_number,
        crop_scale=args.crop_scale,
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        output_all=args.output_all,
        pool_type=args.pool_type,
        attentional_pool=args.attentional_pool,
        add_zero_attn=args.add_zero_attn,
        cosmos=args.cosmos,
        use_adapter=args.use_adapter,
        **model_kwargs,
    )

    # HarMA Adapter: 参数高效微调 - 冻结骨干网络，只训练 Adapter
    def freeze_backbone_train_adapter(model):
        """
        冻结模型骨干网络，只训练 Adapter 相关参数
        """
        # 检查模型是否使用了 Adapter
        has_adapter = False
        for name, _ in model.named_parameters():
            if 'adapter' in name.lower() or 'mmadapter' in name.lower():
                has_adapter = True
                break
        
        if not has_adapter:
            logging.info("No Adapter found in model, skipping parameter freezing.")
            return
        
        logging.info("=" * 80)
        logging.info("HarMA Adapter: Freezing backbone and enabling adapter training")
        logging.info("=" * 80)
        
        # 第一步：冻结所有参数
        total_params = 0
        for param in model.parameters():
            param.requires_grad = False
            total_params += param.numel()
        
        # 第二步：解冻 Adapter 相关参数
        trainable_params = 0
        frozen_params = 0
        trainable_names = []
        
        for name, param in model.named_parameters():
            # 只解冻 Adapter 相关参数（不包括 Backbone 的 LayerNorm）
            if 'adapter' in name.lower() or 'gate_adapter' in name.lower():
                param.requires_grad = True
                trainable_params += param.numel()
                trainable_names.append(name)
            else:
                frozen_params += param.numel()
        
        # 打印统计信息
        logging.info(f"Total parameters:      {total_params:,} ({total_params/1e6:.2f}M)")
        logging.info(f"Trainable parameters:  {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        logging.info(f"Frozen parameters:     {frozen_params:,} ({frozen_params/1e6:.2f}M)")
        logging.info(f"Trainable ratio:       {100*trainable_params/total_params:.2f}%")
        logging.info("=" * 80)
        logging.info(f"Trainable parameter groups ({len(trainable_names)} groups):")
        for name in trainable_names[:10]:  # 只显示前10个
            logging.info(f"  - {name}")
        if len(trainable_names) > 10:
            logging.info(f"  ... and {len(trainable_names) - 10} more")
        logging.info("=" * 80)
    
    # teacher and student start with the same weights
    # IMPORTANT: deepcopy BEFORE freezing, so teacher gets unfrozen params
    import copy
    teacher = copy.deepcopy(student)
    if args.init_last_layer:
        student.init_parameters_last_transformer_layer()
        teacher.init_parameters_last_transformer_layer()

    # 应用参数冻结 (AFTER deepcopy)
    freeze_backbone_train_adapter(student)

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    if args.distill:
        # FIXME: currently assumes the model you're distilling from has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model,
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
        )
    if args.use_bnb_linear is not None:
        print('=> using a layer from bitsandbytes.\n'
              '   this is an experimental feature which requires two extra pip installs\n'
              '   pip install bitsandbytes triton'
              '   please make sure to use triton 2.0.0')
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear
        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(
            bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(student, linear_replacement_cls)
        replace_linear(teacher, linear_replacement_cls)
        student = student.to(device)
        teacher = teacher.to(device)

    random_seed(args.seed, args.rank)

    if args.trace:
        student = trace_model(
            student, batch_size=args.batch_size, device=device)
        teacher = trace_model(
            teacher, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        student.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
        teacher.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        student.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)
        teacher.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)
    if args.grad_checkpointing:
        student.set_grad_checkpointing()
        teacher.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(student)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs_dir, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            student = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student)
            teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        student = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[device], **ddp_args)

        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(
                dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        def exclude(
            n, p): return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n

        def include(n, p): return not exclude(n, p)

        named_parameters = list(student.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(
            n, p) and p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(
                optimizer, named_parameters=student.named_parameters())
            hvd.broadcast_parameters(student.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None

    if args.huggingface_model_name != '':
        # based on huggingface model name, download the pre-trained weights, the downloaded path is passed as the 'resume' arguments
        huggingface_model_name, huggingface_repo_name = args.huggingface_model_name, args.huggingface_repo_name
        args.resume = download_weights_from_hf(model_repo=huggingface_repo_name, filename=huggingface_model_name)
        
    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd_student = checkpoint["student"]
            sd_teacher = checkpoint["teacher"]
            if not args.distributed and next(iter(sd_student.items()))[0].startswith('module'):
                sd_student = {k[len('module.'):]: v for k,
                              v in sd_student.items()}
            if not args.distributed and next(iter(sd_teacher.items()))[0].startswith('module'):
                sd_teacher = {k[len('module.'):]: v for k,
                              v in sd_teacher.items()}

            student.load_state_dict(sd_student)
            teacher.load_state_dict(sd_teacher)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(
                f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            student.load_state_dict(checkpoint["student"])
            teacher.load_state_dict(checkpoint["teacher"])
            logging.info(
                f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    tokenizer = get_tokenizer(args.model)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches //
                       args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None, \
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (
                data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)
        # momentum parameter is increased to 1. during training with a cosine schedule
        momentum_scheduler = cosine_scheduler(
            args.momentum_teacher, 1, warmup_length=0, steps=total_steps)
        
    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs_dir and args.logs_dir.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            if "val" in data:
                args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        if args.resume is not None:
            wand_resume = 'auto'  # wand_resume = 'must'
            wand_name = None
        else:
            wand_resume = None
            wand_name = args.name
        wandb.init(
            entity=args.wandb_entity_name,
            project=args.wandb_project_name,
            name=wand_name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume=wand_resume,
            config=vars(args),
        )
        if args.debug:
            wandb.watch([student, teacher], log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_student = student
    original_teacher = teacher
    if args.torchcompile:
        logging.info('Compiling model...')
        student = torch.compile(original_student)
        teacher = torch.compile(original_teacher)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(student)
        # Evaluate.
        if args.val_data == 'retrieval':
            if args.pretrained in ['laion400m_e32', 'datacomp_xl_s13b_b90k', 'laion2b_s34b_b88k', 'laion400m_e32', 'datacomp_xl_s13b_b90k', 'laion2b_s34b_b79k']:
                zeroshot_evaluate_retrieval(student, None, 'openclip', '', data, start_epoch, args, tokenizer=tokenizer)
            else:
                zeroshot_evaluate_retrieval(student, teacher, 'student', 'teacher', data, start_epoch, args, tokenizer=tokenizer)
        elif args.val_data == 'classification':
            if args.pretrained in ['laion400m_e32', 'datacomp_xl_s13b_b90k', 'laion2b_s34b_b88k', 'laion400m_e32', 'datacomp_xl_s13b_b90k', 'laion2b_s34b_b79k']:
                zeroshot_evaluate_classification(student, None, 'openclip', '', data, start_epoch, args, tokenizer=tokenizer)
            else:
                zeroshot_evaluate_classification(student, teacher, 'student', 'teacher', data, start_epoch, args, tokenizer=tokenizer)
        return

    # evaluate(student, teacher, 'student', 'teacher', data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)

    loss = create_loss(args)

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(student, teacher, data, loss, epoch, optimizer, scaler, scheduler,
                             momentum_scheduler, dist_model, args, tb_writer=writer)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val_coco', 'val', 'imagenet-val', 'imagenet-v2')):
            evaluate(student, teacher, 'student', 'teacher', data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "student": original_student.state_dict(),
                "teacher": original_teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (
                    completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path,
                                 f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(
                    args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(
                    args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs_dir, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs_dir, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path,
             ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
