@echo off
REM 最终优化的GPU训练脚本 - 针对NVIDIA A10

echo =========================================
echo   GPU最终优化训练
echo =========================================

REM 关键修复:
REM 1. num_workers降到8（之前40太多了！）
REM 2. batch_size增大到512（充分利用22GB显存）
REM 3. 禁用persistent_workers（减少内存开销）

python scripts/train_universal_model.py ^
  --model-type lstm ^
  --epochs 30 ^
  --batch-size 512 ^
  --hidden-size 256 ^
  --num-layers 3 ^
  --stock-embedding-dim 64 ^
  --learning-rate 0.001 ^
  --device cuda ^
  --amp ^
  --num-workers 8 ^
  --pin-memory ^
  --output-dir out/universal_model_final

echo.
echo 训练完成！
pause
