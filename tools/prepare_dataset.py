#!/usr/bin/env python3
"""
将自定义数据集转换为webdataset格式
用法: python prepare_dataset.py --image-dir /path/to/images --caption-file /path/to/captions.csv --output-dir /path/to/output
"""

import os
import json
import tarfile
import argparse
from pathlib import Path
from PIL import Image
import io
from tqdm import tqdm
import pandas as pd

def create_webdataset_shards(image_dir, caption_file, output_dir, shard_size=10000):
    """
    将图像和文本描述转换为webdataset格式
    
    Args:
        image_dir: 图像所在目录
        caption_file: CSV文件，包含 image_name, caption 两列
        output_dir: 输出目录
        shard_size: 每个tar文件包含的样本数
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取caption文件
    if caption_file.endswith('.csv'):
        df = pd.read_csv(caption_file)
    elif caption_file.endswith('.json'):
        with open(caption_file, 'r') as f:
            data = json.load(f)
        # 假设JSON格式为 {image_name: caption} 或 [{image_name, caption}, ...]
        if isinstance(data, dict):
            df = pd.DataFrame(list(data.items()), columns=['image_name', 'caption'])
        else:
            df = pd.DataFrame(data)
    else:
        raise ValueError("caption_file must be .csv or .json")
    
    print(f"总共加载 {len(df)} 个样本")
    
    # 创建tar shards
    shard_idx = 0
    sample_idx = 0
    current_tar = None
    current_tar_path = None
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 创建新的shard
        if sample_idx % shard_size == 0:
            if current_tar is not None:
                current_tar.close()
            
            shard_idx = sample_idx // shard_size
            current_tar_path = os.path.join(output_dir, f"dataset-{shard_idx:05d}.tar")
            current_tar = tarfile.open(current_tar_path, "w")
            print(f"创建新的shard: {current_tar_path}")
        
        # 获取图像和caption
        image_name = row['image_name']
        caption = row['caption']
        
        # 构建图像路径
        image_path = os.path.join(image_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"警告: 图像不存在 {image_path}，跳过")
            continue
        
        try:
            # 读取图像
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # 获取图像扩展名
            _, ext = os.path.splitext(image_name)
            
            # 创建tar info
            # 添加图像
            image_tarinfo = tarfile.TarInfo(name=f"{sample_idx:06d}.jpg")
            image_tarinfo.size = len(image_data)
            current_tar.addfile(image_tarinfo, io.BytesIO(image_data))
            
            # 添加caption
            caption_bytes = caption.encode('utf-8')
            caption_tarinfo = tarfile.TarInfo(name=f"{sample_idx:06d}.txt")
            caption_tarinfo.size = len(caption_bytes)
            current_tar.addfile(caption_tarinfo, io.BytesIO(caption_bytes))
            
            # 添加JSON元数据（可选）
            metadata = {
                'image_name': image_name,
                'caption': caption
            }
            metadata_bytes = json.dumps(metadata, ensure_ascii=False).encode('utf-8')
            metadata_tarinfo = tarfile.TarInfo(name=f"{sample_idx:06d}.json")
            metadata_tarinfo.size = len(metadata_bytes)
            current_tar.addfile(metadata_tarinfo, io.BytesIO(metadata_bytes))
            
            sample_idx += 1
            
        except Exception as e:
            print(f"错误处理 {image_path}: {e}")
            continue
    
    # 关闭最后一个tar
    if current_tar is not None:
        current_tar.close()
    
    print(f"完成！共创建 {shard_idx + 1} 个shard，总共 {sample_idx} 个样本")
    print(f"输出目录: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='将自定义数据集转换为webdataset格式')
    parser.add_argument('--image-dir', required=True, help='图像所在目录')
    parser.add_argument('--caption-file', required=True, help='CSV或JSON文件，包含图像名称和描述')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--shard-size', type=int, default=10000, help='每个tar文件包含的样本数')
    
    args = parser.parse_args()
    
    create_webdataset_shards(
        args.image_dir,
        args.caption_file,
        args.output_dir,
        args.shard_size
    )

if __name__ == '__main__':
    main()
