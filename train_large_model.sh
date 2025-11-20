#!/bin/bash
# 大模型训练 - 充分利用NVIDIA A10的22GB显存

echo "========================================="
echo "  大模型GPU训练 - 800万参数"
echo "========================================="

echo ""
echo "配置:"
echo "  模型参数: ~800万个 (当前43万)"
echo "  hidden_size: 512 (当前64)"
echo "  num_layers: 4 (当前2)"
echo "  batch_size: 1024 (当前256)"
echo "  预期显存: 12-18GB"
echo "  预期GPU使用率: 70-90%"
echo ""

python scripts/train_universal_model.py \
  --model-type lstm \
  --epochs 30 \
  --batch-size 1024 \
  --hidden-size 512 \
  --num-layers 4 \
  --stock-embedding-dim 128 \
  --learning-rate 0.001 \
  --device cuda \
  --amp \
  --num-workers 8 \
  --pin-memory \
  --output-dir out/universal_large_model

echo ""
echo "训练完成！"
