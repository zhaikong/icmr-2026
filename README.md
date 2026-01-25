# PECA-CLIP: Parameter-Efficient Cross-modal Adaptation for Vision-Language Pre-training

<p align="center">
  <img src="assets/framework.png" width="800">
</p>

## Overview

PECA-CLIP is a parameter-efficient fine-tuning framework for vision-language models, specifically designed for domain-specific image-text retrieval tasks. Our method combines:

- **Parameter-Efficient Adapter**: Only trains 2.3% of parameters while achieving superior performance
- **Multi-Granularity Crop Augmentation**: Dual-crop strategy for both images and texts
- **Cross-Modal Self-Distillation**: Leverages COSMOS-style knowledge distillation

## Key Results

On the WPIT9K pest image dataset (9,496 images):

| Method | R@1 | R@5 | R@10 |
|--------|-----|-----|------|
| Full Fine-tuning | 40.89% | 71.97% | 83.35% |
| + Adapter | 46.79% | 82.82% | 91.46% |
| + Dual-Crop | 52.27% | 87.99% | 94.84% |
| **PECA-CLIP (Full)** | **53.11%** | **90.52%** | **95.68%** |

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/PECA-CLIP.git
cd PECA-CLIP

# Create conda environment
conda create -n peca-clip python=3.10
conda activate peca-clip

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.8
- GPU: NVIDIA RTX 4090 24GB (or equivalent)

## Dataset Preparation

### 1. Prepare your dataset

Your dataset should have the following structure:
```
your_dataset/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

Each JSON file should contain:
```json
[
  {
    "image": "image_001.jpg",
    "caption": ["description 1", "description 2", "description 3"],
    "image_id": 1,
    "label": 0
  },
  ...
]
```

### 2. Convert to WebDataset format

```bash
python tools/prepare_dataset_advanced.py \
    --image-dir /path/to/images \
    --json-dir /path/to/annotations \
    --output-dir datasets/your_dataset
```

### 3. Verify dataset

```bash
python tools/check_dataset.py --dataset-dir datasets/your_dataset/train
```

## Training

### Quick Start

```bash
cd src
python -m main \
    --model ViT-B-16 \
    --pretrained ../pretrained/cosmos_vitb16_cc3m.pt \
    --use-adapter \
    --cosmos \
    --attentional-pool \
    --output-all \
    --use-imagecrop-aug \
    --global-crops-number 2 \
    --local-crops-number 2 \
    --crop-scale 0.4 \
    --caption-sampling-mode textcrop_pixelprose \
    --num-sampled-captions 4 \
    --train-data '../datasets/your_dataset/train/train-{00000..00007}.tar' \
    --train-num-samples 7596 \
    --val-data '../datasets/your_dataset/val/val-00000.tar' \
    --val-num-samples 949 \
    --dataset-type webdataset \
    --batch-size 32 \
    --epochs 50 \
    --lr 5e-4 \
    --warmup 500 \
    --wd 0.1 \
    --precision amp \
    --momentum-teacher 0.9995 \
    --workers 4 \
    --logs ../logs \
    --name experiment_name
```

### Using Experiment Scripts

We provide pre-configured experiment scripts in the `experiments/` directory:

```bash
# Full method (recommended)
bash experiments/E12_full_v4.sh

# Ablation experiments
bash experiments/E0_no_adapter.sh      # Baseline without adapter
bash experiments/E1_adapter_only.sh    # Adapter only
bash experiments/E9_image_text_crop.sh # Adapter + dual-crop
```

## Model Architecture

### Adapter Module

The adapter is inserted into each transformer block:

```python
class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck_dim=64, dropout=0.1):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + self.gate * x
```

### Multi-Crop Augmentation

- **Global crops**: 2 crops at scale (0.4, 1.0), size 224×224
- **Local crops**: 2 crops at scale (0.05, 0.4), size 96×96
- **Text crops**: 4 sampled text views from long descriptions

## Pre-trained Weights

Download the COSMOS pre-trained weights:

| Model | Dataset | Download |
|-------|---------|----------|
| ViT-B/16 | CC3M | [cosmos_vitb16_cc3m.pt](https://drive.google.com/...) |

Place the weights in `pretrained/` directory.

## Evaluation

```bash
cd src
python -m main \
    --model ViT-B-16 \
    --pretrained ../checkpoints/best_model.pt \
    --val-data '../datasets/your_dataset/test/test-00000.tar' \
    --val-num-samples 951 \
    --dataset-type webdataset \
    --batch-size 32
```

## Project Structure

```
PECA-CLIP/
├── src/
│   ├── main.py                 # Main training script
│   ├── open_clip/              # Model implementations
│   │   ├── adapter.py          # Adapter module
│   │   ├── model.py            # CLIP model with adapter
│   │   ├── loss.py             # COSMOS loss functions
│   │   ├── transform.py        # Data augmentation
│   │   └── ...
│   ├── training/               # Training utilities
│   │   ├── train.py            # Training loop
│   │   ├── data.py             # Data loading
│   │   └── ...
│   └── dataloaders/            # Dataset loaders
├── experiments/                # Experiment scripts
├── tools/                      # Data preparation tools
├── docs/                       # Documentation
└── requirements.txt
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{peca-clip2024,
  title={PECA-CLIP: Parameter-Efficient Cross-modal Adaptation for Vision-Language Pre-training},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Acknowledgements

This project is built upon:
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [COSMOS](https://github.com/xxx/cosmos)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
