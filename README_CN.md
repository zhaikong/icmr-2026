# PECA-CLIP: 参数高效的跨模态适配视觉语言预训练

## 概述

PECA-CLIP是一个参数高效的视觉语言模型微调框架，专为领域特定的图文检索任务设计。

### 核心技术

1. **参数高效适配器**: 仅训练2.3%参数，避免过拟合
2. **多粒度双裁剪增强**: 图像和文本的双重裁剪策略
3. **跨模态自蒸馏**: COSMOS风格的知识蒸馏

## 实验结果

在WPIT9K害虫图像数据集（9,496张图像）上的结果：

| 方法 | R@1 | R@5 | R@10 |
|------|-----|-----|------|
| 全量微调 | 40.89% | 71.97% | 83.35% |
| + 适配器 | 46.79% | 82.82% | 91.46% |
| + 双裁剪 | 52.27% | 87.99% | 94.84% |
| **完整方法** | **53.11%** | **90.52%** | **95.68%** |

## 安装

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/PECA-CLIP.git
cd PECA-CLIP

# 创建conda环境
conda create -n peca-clip python=3.10
conda activate peca-clip

# 安装依赖
pip install -r requirements.txt
```

## 数据准备

### 1. 数据集结构

```
your_dataset/
├── images/
│   ├── image_001.jpg
│   └── ...
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

### 2. 转换为WebDataset格式

```bash
python tools/prepare_dataset_advanced.py \
    --image-dir /path/to/images \
    --json-dir /path/to/annotations \
    --output-dir datasets/your_dataset
```

## 训练

```bash
cd src
python -m main \
    --model ViT-B-16 \
    --pretrained ../pretrained/cosmos_vitb16_cc3m.pt \
    --use-adapter \
    --cosmos \
    --attentional-pool \
    --use-imagecrop-aug \
    --global-crops-number 2 \
    --local-crops-number 2 \
    --caption-sampling-mode textcrop_pixelprose \
    --num-sampled-captions 4 \
    --train-data '../datasets/train/train-{00000..00007}.tar' \
    --train-num-samples 7596 \
    --batch-size 32 \
    --epochs 50 \
    --lr 5e-4
```

## 消融实验

| 组件 | 贡献 |
|------|------|
| 预训练模型 | 基础能力 |
| 适配器 | +5.90% |
| 双裁剪增强 | +5.48% |
| 注意力+蒸馏 | +0.84% |

## 许可证

MIT License
