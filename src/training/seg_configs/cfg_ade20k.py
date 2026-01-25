# ADE20K config file that define test pipeline
# Please refer to here https://github.com/wangf3014/SCLIP/blob/main/configs/cfg_ade20k.py
_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./training/seg_configs/cls_ade20k.txt'
)

# dataset settings
dataset_type = 'ADE20KDataset'
data_root = '/mmsegmentation_datasets/data/ade/ADEChallengeData2016'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 336), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))