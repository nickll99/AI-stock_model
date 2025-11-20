# 🚀 GPU训练快速修复

## 问题
你的GPU使用率只有1.64%，显存只用了0.4GB/22.5GB，训练速度很慢。

## 原因
当前命令缺少关键优化参数：
```bash
# 你当前的命令 ❌
python scripts/train_universal_model.py \
  --model-type lstm \
  --epochs 30 \
  --batch-size 64 \      # 太小
  --hidden-size 64 \     # 太小
  --stock-embedding-dim 16  # 太小
# 缺少 --amp, --num-workers 等关键参数
```

## 解决方案

### 立即执行（推荐）

**停止当前训练** (Ctrl+C)，然后运行：

```bash
python scripts/train_universal_model.py \
  --model-type lstm \
  --epochs 30 \
  --batch-size 256 \
  --hidden-size 256 \
  --num-layers 3 \
  --stock-embedding-dim 64 \
  --device cuda \
  --amp \
  --num-workers 8 \
  --pin-memory
```

### 预期效果
- ✅ GPU使用率: 1.64% → **60-90%**
- ✅ 显存使用: 0.4GB → **8-12GB**
- ✅ 训练速度: **提升5-10倍**
- ✅ 每轮时间: 5分钟 → **30-60秒**

## 关键改动

| 参数 | 之前 | 现在 | 说明 |
|------|------|------|------|
| `--batch-size` | 64 | **256** | 增大4倍，充分利用GPU并行 |
| `--hidden-size` | 64 | **256** | 增大4倍，增加模型复杂度 |
| `--stock-embedding-dim` | 16 | **64** | 增大4倍，更好的股票表示 |
| `--amp` | ❌ | **✅** | 启用混合精度，速度提升2-3倍 |
| `--num-workers` | 4 | **8** | 加快数据加载 |
| `--pin-memory` | ❌ | **✅** | 加速CPU→GPU传输 |

## 快速验证

运行诊断工具：
```bash
python diagnose_gpu_usage.py
```

## 备选方案

如果显存不足，使用保守配置：
```bash
python scripts/train_universal_model.py \
  --model-type lstm \
  --epochs 30 \
  --batch-size 128 \
  --hidden-size 128 \
  --stock-embedding-dim 32 \
  --device cuda \
  --amp \
  --num-workers 6 \
  --pin-memory
```

## 监控GPU

训练时另开一个终端：
```bash
# Windows PowerShell
while($true) { nvidia-smi; sleep 1; cls }

# Linux
watch -n 1 nvidia-smi
```

应该看到：
- GPU使用率: 60-90%
- 显存使用: 8-12GB
- 温度: 60-80°C

---

**立即行动：** 停止当前训练，使用上面的优化命令重新开始！
