#!/usr/bin/env python3
"""
将img数据集转换为WebDataset格式的专用脚本
针对RTX 4090优化
"""

import os
import json
import tarfile
import argparse
from pathlib import Path
from PIL import Image
import io
from tqdm import tqdm
import random

def create_webdataset_from_json(image_dir, json_file, output_dir, split_name, shard_size=1000):
    """
    从JSON文件创建WebDataset格式的tar文件
    
    Args:
        image_dir: 图像所在目录
        json_file: JSON标注文件
        output_dir: 输出目录
        split_name: 数据集分割名称（train/val/test）
        shard_size: 每个tar文件包含的样本数
    """
    
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"处理 {split_name} 集: {len(data)} 个样本")
    
    # 创建输出目录
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    shard_idx = 0
    sample_idx = 0
    current_tar = None
    current_tar_path = None
    
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'errors': []
    }
    
    for idx, item in tqdm(enumerate(data), total=len(data), desc=f"处理 {split_name} 集"):
        # 创建新的shard
        if sample_idx % shard_size == 0:
            if current_tar is not None:
                current_tar.close()
            
            shard_idx = sample_idx // shard_size
            current_tar_path = os.path.join(split_dir, f"{split_name}-{shard_idx:05d}.tar")
            current_tar = tarfile.open(current_tar_path, "w")
        
        # 获取图像和caption信息
        image_name = item['image']
        captions = item['caption']  # 这是一个列表
        image_id = item.get('image_id', idx)
        label = item.get('label', 0)
        
        image_path = os.path.join(image_dir, image_name)
        
        stats['total'] += 1
        
        try:
            # 检查图像是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像不存在: {image_path}")
            
            # 读取图像
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # 验证图像
            try:
                Image.open(io.BytesIO(image_data)).verify()
            except Exception as e:
                raise ValueError(f"无效的图像: {e}")
            
            # 合并所有caption为长描述（COSMOS需要长描述）
            combined_caption = " ".join(captions)
            
            # 创建tar info
            # 添加图像
            image_tarinfo = tarfile.TarInfo(name=f"{sample_idx:06d}.jpg")
            image_tarinfo.size = len(image_data)
            current_tar.addfile(image_tarinfo, io.BytesIO(image_data))
            
            # 添加合并后的长描述
            caption_bytes = combined_caption.encode('utf-8')
            caption_tarinfo = tarfile.TarInfo(name=f"{sample_idx:06d}.txt")
            caption_tarinfo.size = len(caption_bytes)
            current_tar.addfile(caption_tarinfo, io.BytesIO(caption_bytes))
            
            # 添加JSON元数据（包含所有原始信息）
            metadata = {
                'image_name': image_name,
                'combined_caption': combined_caption,
                'original_captions': captions,
                'image_id': image_id,
                'label': label,
                'sample_id': sample_idx
            }
            metadata_bytes = json.dumps(metadata, ensure_ascii=False).encode('utf-8')
            metadata_tarinfo = tarfile.TarInfo(name=f"{sample_idx:06d}.json")
            metadata_tarinfo.size = len(metadata_bytes)
            current_tar.addfile(metadata_tarinfo, io.BytesIO(metadata_bytes))
            
            sample_idx += 1
            stats['success'] += 1
            
        except Exception as e:
            stats['failed'] += 1
            error_msg = f"错误处理 {image_path}: {e}"
            stats['errors'].append(error_msg)
            if len(stats['errors']) <= 10:  # 只打印前10个错误
                print(f"  {error_msg}")
    
    # 关闭最后一个tar
    if current_tar is not None:
        current_tar.close()
    
    print(f"\n{split_name} 集统计:")
    print(f"  总数: {stats['total']}")
    print(f"  成功: {stats['success']}")
    print(f"  失败: {stats['failed']}")
    print(f"  生成的tar文件数: {shard_idx + 1}")
    
    return stats

def main():
    # 固定路径配置
    base_dir = "/home/vision/mwk/cosmos/cosmos/cosmos-main/img"
    image_dir = os.path.join(base_dir, "img")
    json_dir = os.path.join(base_dir, "json")
    output_dir = "/home/vision/mwk/cosmos/cosmos/cosmos-main/datasets/img_dataset"
    
    print("=" * 60)
    print("IMG数据集 -> WebDataset 转换")
    print("=" * 60)
    
    # 检查输入目录
    if not os.path.exists(image_dir):
        print(f"错误: 图像目录不存在 {image_dir}")
        return
    
    if not os.path.exists(json_dir):
        print(f"错误: JSON目录不存在 {json_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换各个数据集
    splits = ['train', 'val', 'test']
    all_stats = {}
    
    for split in splits:
        json_file = os.path.join(json_dir, f"{split}.json")
        if os.path.exists(json_file):
            print(f"\n转换 {split} 集...")
            stats = create_webdataset_from_json(
                image_dir=image_dir,
                json_file=json_file,
                output_dir=output_dir,
                split_name=split,
                shard_size=1000  # RTX 4090适合的批次大小
            )
            all_stats[split] = stats
        else:
            print(f"警告: {json_file} 不存在，跳过")
    
    # 总结
    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)
    
    total_samples = 0
    total_success = 0
    total_failed = 0
    
    for split, stats in all_stats.items():
        print(f"{split.upper()} 集:")
        print(f"  成功: {stats['success']}")
        print(f"  失败: {stats['failed']}")
        total_samples += stats['total']
        total_success += stats['success']
        total_failed += stats['failed']
    
    print(f"\n总计:")
    print(f"  总样本数: {total_samples}")
    print(f"  成功转换: {total_success}")
    print(f"  转换失败: {total_failed}")
    print(f"  成功率: {total_success/total_samples*100:.1f}%")
    
    print(f"\n输出目录: {output_dir}")
    print("下一步: 运行训练脚本")

if __name__ == "__main__":
    main()
