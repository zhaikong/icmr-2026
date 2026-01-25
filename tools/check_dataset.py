#!/usr/bin/env python3
"""
数据集检查工具
用于验证WebDataset格式的tar文件是否正确
"""

import os
import tarfile
import json
import argparse
from pathlib import Path
from PIL import Image
import io
from tqdm import tqdm
from collections import defaultdict

def check_tar_file(tar_path, sample_limit=None):
    """检查单个tar文件"""
    stats = {
        'total_files': 0,
        'images': 0,
        'captions': 0,
        'metadata': 0,
        'other': 0,
        'errors': [],
        'samples': []
    }
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()
            stats['total_files'] = len(members)
            
            # 按样本ID分组
            samples_dict = defaultdict(dict)
            
            for member in members:
                if member.isfile():
                    name = member.name
                    
                    # 提取样本ID
                    parts = name.split('.')
                    if len(parts) >= 2:
                        sample_id = parts[0]
                        ext = parts[-1]
                        
                        if ext == 'jpg' or ext == 'png':
                            stats['images'] += 1
                            samples_dict[sample_id]['image'] = name
                        elif ext == 'txt':
                            stats['captions'] += 1
                            samples_dict[sample_id]['caption'] = name
                        elif ext == 'json':
                            stats['metadata'] += 1
                            samples_dict[sample_id]['metadata'] = name
                        else:
                            stats['other'] += 1
            
            # 验证样本完整性
            sample_count = 0
            for sample_id, files in sorted(samples_dict.items()):
                if sample_limit and sample_count >= sample_limit:
                    break
                
                sample_info = {'id': sample_id, 'files': files}
                
                # 检查是否有图像和描述
                if 'image' not in files:
                    sample_info['error'] = '缺少图像'
                    stats['errors'].append(f"样本 {sample_id}: 缺少图像")
                
                if 'caption' not in files:
                    sample_info['error'] = '缺少描述'
                    stats['errors'].append(f"样本 {sample_id}: 缺少描述")
                
                # 尝试读取和验证内容
                try:
                    if 'image' in files:
                        image_member = tar.getmember(files['image'])
                        image_data = tar.extractfile(image_member).read()
                        Image.open(io.BytesIO(image_data)).verify()
                        sample_info['image_size'] = len(image_data)
                    
                    if 'caption' in files:
                        caption_member = tar.getmember(files['caption'])
                        caption_data = tar.extractfile(caption_member).read()
                        caption_text = caption_data.decode('utf-8')
                        sample_info['caption_length'] = len(caption_text)
                    
                    if 'metadata' in files:
                        metadata_member = tar.getmember(files['metadata'])
                        metadata_data = tar.extractfile(metadata_member).read()
                        metadata = json.loads(metadata_data.decode('utf-8'))
                        sample_info['metadata'] = metadata
                
                except Exception as e:
                    sample_info['error'] = str(e)
                    stats['errors'].append(f"样本 {sample_id}: {e}")
                
                stats['samples'].append(sample_info)
                sample_count += 1
    
    except Exception as e:
        stats['errors'].append(f"无法打开tar文件: {e}")
    
    return stats

def check_dataset_directory(dataset_dir, sample_limit=5):
    """检查整个数据集目录"""
    print(f"检查数据集目录: {dataset_dir}")
    print("=" * 80)
    
    tar_files = sorted(Path(dataset_dir).glob("*.tar"))
    
    if not tar_files:
        print("警告: 未找到tar文件！")
        return
    
    print(f"找到 {len(tar_files)} 个tar文件\n")
    
    total_stats = {
        'total_tar_files': len(tar_files),
        'total_samples': 0,
        'total_images': 0,
        'total_captions': 0,
        'total_metadata': 0,
        'total_errors': 0,
        'tar_files_info': []
    }
    
    for tar_path in tqdm(tar_files, desc="检查tar文件"):
        stats = check_tar_file(tar_path, sample_limit=sample_limit)
        
        tar_info = {
            'name': tar_path.name,
            'path': str(tar_path),
            'total_files': stats['total_files'],
            'images': stats['images'],
            'captions': stats['captions'],
            'metadata': stats['metadata'],
            'errors': len(stats['errors']),
            'samples': len(stats['samples'])
        }
        
        total_stats['tar_files_info'].append(tar_info)
        total_stats['total_samples'] += len(stats['samples'])
        total_stats['total_images'] += stats['images']
        total_stats['total_captions'] += stats['captions']
        total_stats['total_metadata'] += stats['metadata']
        total_stats['total_errors'] += len(stats['errors'])
        
        # 打印单个tar文件的信息
        print(f"\n文件: {tar_path.name}")
        print(f"  总文件数: {stats['total_files']}")
        print(f"  图像数: {stats['images']}")
        print(f"  描述数: {stats['captions']}")
        print(f"  元数据数: {stats['metadata']}")
        print(f"  样本数: {len(stats['samples'])}")
        
        if stats['errors']:
            print(f"  错误数: {len(stats['errors'])}")
            for error in stats['errors'][:5]:  # 只显示前5个错误
                print(f"    - {error}")
            if len(stats['errors']) > 5:
                print(f"    ... 还有 {len(stats['errors']) - 5} 个错误")
        
        # 显示样本信息
        if stats['samples']:
            print(f"  样本示例:")
            for sample in stats['samples'][:3]:  # 显示前3个样本
                print(f"    - ID: {sample['id']}")
                if 'image_size' in sample:
                    print(f"      图像大小: {sample['image_size']} bytes")
                if 'caption_length' in sample:
                    print(f"      描述长度: {sample['caption_length']} 字符")
                if 'error' in sample:
                    print(f"      错误: {sample['error']}")
    
    # 打印总体统计
    print("\n" + "=" * 80)
    print("总体统计:")
    print(f"  Tar文件数: {total_stats['total_tar_files']}")
    print(f"  总样本数: {total_stats['total_samples']}")
    print(f"  总图像数: {total_stats['total_images']}")
    print(f"  总描述数: {total_stats['total_captions']}")
    print(f"  总元数据数: {total_stats['total_metadata']}")
    print(f"  总错误数: {total_stats['total_errors']}")
    
    # 验证完整性
    print("\n完整性检查:")
    if total_stats['total_images'] == total_stats['total_samples']:
        print("  ✓ 每个样本都有图像")
    else:
        print(f"  ✗ 图像数量不匹配: {total_stats['total_images']} vs {total_stats['total_samples']}")
    
    if total_stats['total_captions'] == total_stats['total_samples']:
        print("  ✓ 每个样本都有描述")
    else:
        print(f"  ✗ 描述数量不匹配: {total_stats['total_captions']} vs {total_stats['total_samples']}")
    
    if total_stats['total_errors'] == 0:
        print("  ✓ 没有错误")
    else:
        print(f"  ✗ 发现 {total_stats['total_errors']} 个错误")
    
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='检查WebDataset格式的数据集')
    parser.add_argument('--dataset-dir', required=True, help='数据集目录')
    parser.add_argument('--sample-limit', type=int, default=5, 
                       help='每个tar文件检查的样本数限制')
    
    args = parser.parse_args()
    
    check_dataset_directory(args.dataset_dir, sample_limit=args.sample_limit)

if __name__ == '__main__':
    main()
