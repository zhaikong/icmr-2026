#!/bin/bash

################################################################################
# 消融实验 E12-v4: 完整方法（在cosmos环境中运行）
# 
# 配置（全部启用）：
# ✅ HarMA Adapter
# ✅ 图像多裁剪 (2全局 + 2局部)
# ✅ 文本裁切 (textcrop_pixelprose)
# ✅ 交叉注意力 (attentional-pool)
# ✅ COSMOS自蒸馏
################################################################################

echo "========================================"
echo "开始消融实验 E12-v4"
echo "配置: 完整方法（全部功能）"
echo "目标: 复现53.11%的结果"
echo "========================================"
echo ""

# 激活cosmos环境
echo "激活cosmos环境..."
source /home/mc02229/ENTER/bin/activate
echo "✅ 当前Python: $(which python)"
echo "✅ 当前环境: $VIRTUAL_ENV"
echo ""

# 进入src目录
cd /home/vision/mwk/cosmos+harma/cosmos/cosmos/cosmos-main/src

# 检查预训练权重
PRETRAINED="../cosmos_vitb16_cc3m.pt"
if [ ! -f "$PRETRAINED" ]; then
    echo "❌ 错误: 预训练权重不存在: $PRETRAINED"
    exit 1
fi
echo "✅ 预训练权重: $PRETRAINED"

# 检查数据集
TRAIN_DATA="../datasets/img_dataset_fixed/train/train-00000.tar"
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ 错误: 训练数据不存在"
    exit 1
fi
echo "✅ 训练数据: img_dataset_fixed"
echo ""

# 显示配置
echo "实验配置:"
echo "  - HarMA Adapter: ✅"
echo "  - 图像多裁剪: ✅ (2全局 + 2局部)"
echo "  - 文本裁切: ✅ (textcrop_pixelprose, 4个采样)"
echo "  - 交叉注意力: ✅ (attentional-pool)"
echo "  - COSMOS自蒸馏: ✅"
echo "  - 随机种子: 2024"
echo ""
echo "开始训练..."
echo "========================================"
echo ""

# 设置环境变量优化显存使用
export PYTORCH_ALLOC_CONF=expandable_segments:True

python -m main \
    --model ViT-B-16 \
    --pretrained "$PRETRAINED" \
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
    --train-data '../datasets/img_dataset_fixed/train/train-{00000..00007}.tar' \
    --train-num-samples 7596 \
    --val-data '../datasets/img_dataset_fixed/val/val-00000.tar' \
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
    --logs ../logs/ablation/E12_full_v4 \
    --name E12_full_v4_$(date +%Y%m%d_%H%M%S) \
    --save-frequency 10 \
    --seed 2024

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 实验 E12-v4 训练完成!"
    echo "结果保存在: logs/ablation/E12_full_v4/"
else
    echo "❌ 实验 E12-v4 训练失败 (退出码: $EXIT_CODE)"
fi
echo "========================================"

exit $EXIT_CODE
