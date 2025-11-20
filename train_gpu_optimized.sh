#!/bin/bash
# GPU优化训练脚本 - 充分利用GPU性能

echo "========================================="
echo "  GPU优化训练 - 通用模型"
echo "========================================="

# 检查GPU
echo ""
echo "检查GPU状态..."
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}'); print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}'); print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB' if torch.cuda.is_available() else '')"

echo ""
echo "开始训练..."
echo ""

# GPU优化参数说明:
# --amp: 启用混合精度训练，提升速度并减少显存占用
# --batch-size 256: 增大批次大小，充分利用GPU并行能力
# --num-workers 8: 增加数据加载线程，避免GPU等待数据
# --pin-memory: 使用锁页内存，加速CPU到GPU的数据传输
# --hidden-size 128: 适中的隐藏层大小，平衡性能和精度
# --stock-embedding-dim 32: 股票嵌入维度

python scripts/train_universal_model.py \
  --model-type lstm \
  --epochs 30 \
  --batch-size 256 \
  --hidden-size 128 \
  --num-layers 2 \
  --stock-embedding-dim 32 \
  --learning-rate 0.001 \
  --device cuda \
  --amp \
  --num-workers 8 \
  --pin-memory \
  --output-dir out/universal_model_gpu

echo ""
echo "训练完成！"
