#!/usr/bin/env python3
"""
高级数据集准备脚本
支持多种输入格式和数据验证
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
import hashlib
from collections import defaultdict

class DatasetPreparer:
    def __init__(self, image_dir, caption_file, output_dir, shard_size=10000, 
                 train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        """
        初始化数据集准备器
        
        Args:
            image_dir: 图像目录
            caption_file: 描述文件（CSV或JSON）
            output_dir: 输出目录
            shard_size: 每个tar文件的样本数
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
        """
        self.image_dir = image_dir
        self.caption_file = caption_file
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # 验证比例
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            f"比例之和必须为1.0，当前为 {train_ratio + val_ratio + test_ratio}"
        
        os.makedirs(output_dir, exist_ok=True)
        
    def load_captions(self):
        """加载描述文件"""
        print(f"加载描述文件: {self.caption_file}")
        
        if self.caption_file.endswith('.csv'):
            df = pd.read_csv(self.caption_file)
        elif self.caption_file.endswith('.json'):
            with open(self.caption_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                df = pd.DataFrame(list(data.items()), columns=['image_name', 'caption'])
            else:
                df = pd.DataFrame(data)
        else:
            raise ValueError("描述文件必须是 .csv 或 .json 格式")
        
        print(f"加载了 {len(df)} 个样本")
        return df
    
    def validate_dataset(self, df):
        """验证数据集"""
        print("\n验证数据集...")
        
        missing_images = []
        invalid_images = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="验证图像"):
            image_name = row['image_name']
            image_path = os.path.join(self.image_dir, image_name)
            
            if not os.path.exists(image_path):
                missing_images.append(image_name)
                continue
            
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception as e:
                invalid_images.append((image_name, str(e)))
        
        print(f"\n验证结果:")
        print(f"  总样本数: {len(df)}")
        print(f"  缺失图像: {len(missing_images)}")
        print(f"  无效图像: {len(invalid_images)}")
        print(f"  有效样本: {len(df) - len(missing_images) - len(invalid_images)}")
        
        if missing_images:
            print(f"\n缺失的图像 (前10个):")
            for img in missing_images[:10]:
                print(f"  - {img}")
        
        if invalid_images:
            print(f"\n无效的图像 (前10个):")
            for img, err in invalid_images[:10]:
                print(f"  - {img}: {err}")
        
        # 过滤有效样本
        valid_indices = []
        for idx, row in df.iterrows():
            image_name = row['image_name']
            image_path = os.path.join(self.image_dir, image_name)
            
            if os.path.exists(image_path):
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                    valid_indices.append(idx)
                except:
                    pass
        
        df_valid = df.loc[valid_indices].reset_index(drop=True)
        print(f"\n保留 {len(df_valid)} 个有效样本用于训练")
        
        return df_valid
    
    def split_dataset(self, df):
        """分割训练/验证/测试集"""
        print("\n分割数据集...")
        
        import numpy as np
        np.random.seed(self.seed)
        
        n = len(df)
        indices = np.random.permutation(n)
        
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        df_train = df.iloc[train_indices].reset_index(drop=True)
        df_val = df.iloc[val_indices].reset_index(drop=True)
        df_test = df.iloc[test_indices].reset_index(drop=True)
        
        print(f"  训练集: {len(df_train)} 样本")
        print(f"  验证集: {len(df_val)} 样本")
        print(f"  测试集: {len(df_test)} 样本")
        
        return df_train, df_val, df_test
    
    def create_webdataset_shards(self, df, split_name):
        """创建WebDataset格式的tar文件"""
        print(f"\n创建 {split_name} 集的 shards...")
        
        split_dir = os.path.join(self.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        shard_idx = 0
        sample_idx = 0
        current_tar = None
        current_tar_path = None
        
        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'errors': defaultdict(int)
        }
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"处理 {split_name} 集"):
            # 创建新的shard
            if sample_idx % self.shard_size == 0:
                if current_tar is not None:
                    current_tar.close()
                
                shard_idx = sample_idx // self.shard_size
                current_tar_path = os.path.join(split_dir, f"{split_name}-{shard_idx:05d}.tar")
                current_tar = tarfile.open(current_tar_path, "w")
            
            # 获取图像和caption
            image_name = row['image_name']
            caption = row['caption']
            
            image_path = os.path.join(self.image_dir, image_name)
            
            stats['total'] += 1
            
            try:
                # 读取图像
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                
                # 验证图像
                try:
                    Image.open(io.BytesIO(image_data)).verify()
                except Exception as e:
                    raise ValueError(f"无效的图像: {e}")
                
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
                
                # 添加JSON元数据
                metadata = {
                    'image_name': image_name,
                    'caption': caption,
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
                error_type = type(e).__name__
                stats['errors'][error_type] += 1
                if stats['failed'] <= 10:  # 只打印前10个错误
                    print(f"  错误处理 {image_path}: {e}")
        
        # 关闭最后一个tar
        if current_tar is not None:
            current_tar.close()
        
        print(f"\n{split_name} 集统计:")
        print(f"  总数: {stats['total']}")
        print(f"  成功: {stats['success']}")
        print(f"  失败: {stats['failed']}")
        if stats['errors']:
            print(f"  错误类型: {dict(stats['errors'])}")
        
        return stats
    
    def prepare(self, validate=True, split=True):
        """执行完整的数据准备流程"""
        print("=" * 60)
        print("COSMOS 数据集准备")
        print("=" * 60)
        
        # 加载描述
        df = self.load_captions()
        
        # 验证数据集
        if validate:
            df = self.validate_dataset(df)
        
        # 分割数据集
        if split:
            df_train, df_val, df_test = self.split_dataset(df)
            
            # 创建shards
            self.create_webdataset_shards(df_train, 'train')
            self.create_webdataset_shards(df_val, 'val')
            if len(df_test) > 0:
                self.create_webdataset_shards(df_test, 'test')
        else:
            self.create_webdataset_shards(df, 'dataset')
        
        print("\n" + "=" * 60)
        print("数据准备完成！")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)
        
        # 生成配置文件
        self.generate_config()
    
    def generate_config(self):
        """生成训练配置文件"""
        config = {
            'dataset_info': {
                'image_dir': self.image_dir,
                'caption_file': self.caption_file,
                'output_dir': self.output_dir,
                'shard_size': self.shard_size
            },
            'split_info': {
                'train_ratio': self.train_ratio,
                'val_ratio': self.val_ratio,
                'test_ratio': self.test_ratio
            }
        }
        
        config_path = os.path.join(self.output_dir, 'dataset_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"配置文件已保存: {config_path}")

def main():
    parser = argparse.ArgumentParser(description='COSMOS 高级数据集准备工具')
    parser.add_argument('--image-dir', required=True, help='图像所在目录')
    parser.add_argument('--caption-file', required=True, help='CSV或JSON格式的描述文件')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--shard-size', type=int, default=10000, help='每个tar文件的样本数')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--no-validate', action='store_true', help='跳过数据验证')
    parser.add_argument('--no-split', action='store_true', help='不分割数据集')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer(
        image_dir=args.image_dir,
        caption_file=args.caption_file,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    preparer.prepare(
        validate=not args.no_validate,
        split=not args.no_split
    )

if __name__ == '__main__':
    main()
