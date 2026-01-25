#!/bin/bash

################################################################################
# 消融实验 E0: 不使用Adapter（零增强基线）
# 
# 目的：验证预训练模型的Zero-shot能力或最小微调效果
# 
# 配置：
# ❌ HarMA Adapter（不使用）
# ❌ 图像多裁剪
# ❌ 文本裁切
# ❌ 交叉注意力
# ❌ COSMOS自蒸馏
# 
# 说明：只微调投影层，或完全冻结（看效果选择）
################################################################################

echo "========================================"
echo "开始消融实验 E0"
echo "配置: 不使用Adapter（零增强基线）"
echo "========================================"
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
echo "  - HarMA Adapter: ❌ (不使用)"
echo "  - 图像多裁剪: ❌"
echo "  - 文本裁切: ❌"
echo "  - 交叉注意力: ❌"
echo "  - COSMOS自蒸馏: ❌"
echo ""
echo "说明: 验证预训练模型的基础迁移能力"
echo "      只训练投影层或完全Zero-shot"
echo ""

echo "开始训练..."
echo "========================================"
echo ""

# 设置环境变量优化显存使用
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 不使用--use-adapter
python -m main \
    --model ViT-B-16 \
    --pretrained "$PRETRAINED" \
    --train-data '../datasets/img_dataset_fixed/train/train-{00000..00007}.tar' \
    --train-num-samples 7596 \
    --val-data '../datasets/img_dataset_fixed/val/val-00000.tar' \
    --val-num-samples 949 \
    --dataset-type webdataset \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --warmup 500 \
    --wd 0.1 \
    --precision amp \
    --workers 4 \
    --logs ../logs/ablation/E0_no_adapter \
    --name E0_no_adapter_$(date +%Y%m%d_%H%M%S) \
    --save-frequency 10 \
    --seed 42

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 实验 E0 训练完成!"
    echo "结果保存在: logs/ablation/E0_no_adapter/"
else
    echo "❌ 实验 E0 训练失败 (退出码: $EXIT_CODE)"
fi
echo "========================================"

exit $EXIT_CODE

