# GPU训练优化完整指南

## 🚀 快速开始

### 1. 诊断GPU状态
```bash
python diagnose_gpu_usage.py
```

### 2. 使用优化脚本训练
```bash
# Windows
train_gpu_optimized.bat

# Linux/Mac
bash train_gpu_optimized.sh
```

## 📊 你的问题分析

根据你提供的信息：
- **CPU使用率**: 3.19% (20核) - 太低，说明CPU没有充分准备数据
- **内存使用**: 34.76% (116GB) - 正常
- **GPU使用率**: 1.64% - **太低！GPU几乎空闲**
- **显存使用**: 0.4GB / 22.5GB - **显存利用率不到2%**

### 问题根源

你当前的训练命令：
```bash
python scripts/train_universal_model.py \
  --model-type lstm \
  --epochs 30 \
  --batch-size 64 \
  --hidden-size 64 \
  --stock-embedding-dim 16
```

**存在的问题：**

1. ❌ **未启用混合精度训练** (`--amp`)
2. ❌ **batch_size太小** (64) - GPU没有足够的并行任务
3. ❌ **hidden_size太小** (64) - 模型太简单，GPU算力浪费
4. ❌ **num_workers默认值** (4) - 数据加载速度跟不上GPU
5. ❌ **stock_embedding_dim太小** (16) - 模型参数量不足

## ✅ 优化方案

### 方案1: 激进优化（推荐）

充分利用你的22.5GB显存：

```bash
python scripts/train_universal_model.py \
  --model-type lstm \
  --epochs 30 \
  --batch-size 256 \
  --hidden-size 256 \
  --num-layers 3 \
  --stock-embedding-dim 64 \
  --learning-rate 0.001 \
  --device cuda \
  --amp \
  --num-workers 8 \
  --pin-memory \
  --output-dir out/universal_model_optimized
```

**预期效果：**
- GPU使用率: 60-90%
- 显存使用: 8-12GB
- 训练速度: 提升5-10倍

### 方案2: 保守优化

如果方案1显存不足，使用这个：

```bash
python scripts/train_universal_model.py \
  --model-type lstm \
  --epochs 30 \
  --batch-size 128 \
  --hidden-size 128 \
  --num-layers 2 \
  --stock-embedding-dim 32 \
  --learning-rate 0.001 \
  --device cuda \
  --amp \
  --num-workers 6 \
  --pin-memory \
  --output-dir out/universal_model_balanced
```

**预期效果：**
- GPU使用率: 40-70%
- 显存使用: 4-6GB
- 训练速度: 提升3-5倍

### 方案3: 极限优化

如果你想榨干GPU性能：

```bash
python scripts/train_universal_model.py \
  --model-type lstm \
  --epochs 30 \
  --batch-size 512 \
  --hidden-size 512 \
  --num-layers 4 \
  --stock-embedding-dim 128 \
  --learning-rate 0.001 \
  --device cuda \
  --amp \
  --num-workers 12 \
  --pin-memory \
  --gradient-accumulation-steps 2 \
  --output-dir out/universal_model_extreme
```

**预期效果：**
- GPU使用率: 80-95%
- 显存使用: 15-20GB
- 训练速度: 提升10-15倍

## 🔧 关键参数说明

### 1. `--amp` (混合精度训练)
- **作用**: 使用FP16代替FP32，速度提升2-3倍，显存减少50%
- **必须启用**: 是
- **适用场景**: 所有GPU训练

### 2. `--batch-size`
- **作用**: 每批处理的样本数，越大GPU利用率越高
- **推荐值**: 
  - 小显存(<8GB): 64-128
  - 中显存(8-16GB): 128-256
  - 大显存(>16GB): 256-512
- **你的情况**: 256-512

### 3. `--hidden-size`
- **作用**: LSTM隐藏层大小，影响模型复杂度
- **推荐值**:
  - 快速测试: 64-128
  - 正常训练: 128-256
  - 高精度: 256-512
- **你的情况**: 256-512

### 4. `--num-workers`
- **作用**: 数据加载的并行进程数
- **推荐值**: CPU核心数的1/2到1/3
- **你的情况**: 6-10 (你有20核CPU)

### 5. `--stock-embedding-dim`
- **作用**: 股票嵌入向量维度
- **推荐值**: 32-128
- **你的情况**: 64-128

### 6. `--pin-memory`
- **作用**: 使用锁页内存，加速CPU到GPU数据传输
- **必须启用**: 是
- **性能提升**: 10-30%

## 📈 性能对比

### 当前配置 vs 优化配置

| 指标 | 当前配置 | 优化配置 | 提升 |
|------|---------|---------|------|
| batch_size | 64 | 256 | 4x |
| hidden_size | 64 | 256 | 4x |
| 混合精度 | ❌ | ✅ | 2-3x |
| num_workers | 4 | 8 | 2x |
| GPU使用率 | 1.64% | 60-90% | 50x |
| 显存使用 | 0.4GB | 8-12GB | 25x |
| **训练速度** | **基准** | **5-10x** | **🚀** |

## 🎯 实战步骤

### 步骤1: 诊断当前状态
```bash
python diagnose_gpu_usage.py
```

### 步骤2: 停止当前训练
按 `Ctrl+C` 停止当前训练

### 步骤3: 使用优化命令重新训练

**Windows:**
```bash
train_gpu_optimized.bat
```

**Linux/Mac:**
```bash
bash train_gpu_optimized.sh
```

**或者直接运行:**
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

### 步骤4: 监控GPU使用

**实时监控:**
```bash
# Linux
watch -n 1 nvidia-smi

# Windows (PowerShell)
while($true) { nvidia-smi; sleep 1; cls }
```

**预期看到:**
- GPU使用率: 60-90%
- 显存使用: 8-12GB
- 温度: 60-80°C
- 功耗: 接近TDP

## 🐛 常见问题

### Q1: 显存不足 (Out of Memory)
**解决方案:**
1. 减小 `--batch-size` (256 → 128 → 64)
2. 减小 `--hidden-size` (256 → 128)
3. 减少 `--num-layers` (3 → 2)
4. 启用梯度累积: `--gradient-accumulation-steps 2`

### Q2: GPU使用率仍然很低
**可能原因:**
1. 数据加载太慢 → 增加 `--num-workers`
2. batch_size太小 → 增大 `--batch-size`
3. 模型太简单 → 增大 `--hidden-size`
4. 未启用混合精度 → 添加 `--amp`

### Q3: 训练速度没有提升
**检查清单:**
- [ ] 确认使用了 `--amp`
- [ ] 确认 `--batch-size >= 128`
- [ ] 确认 `--num-workers >= 4`
- [ ] 确认 `--device cuda`
- [ ] 确认数据已缓存到 `data/features/`

### Q4: 数据加载慢
**解决方案:**
1. 确保使用缓存数据（默认启用）
2. 增加 `--num-workers` (8-12)
3. 启用 `--pin-memory`
4. 使用SSD存储缓存数据

## 📊 监控指标

### 理想状态
```
GPU使用率: 70-90%
显存使用: 50-80% (11-18GB / 22.5GB)
CPU使用率: 20-40% (数据加载)
训练速度: 每轮 < 60秒
```

### 当前状态（需要优化）
```
GPU使用率: 1.64% ❌
显存使用: 1.8% (0.4GB / 22.5GB) ❌
CPU使用率: 3.19% ❌
训练速度: 每轮 > 300秒 ❌
```

## 🎓 进阶优化

### 1. 使用Transformer模型
```bash
python scripts/train_universal_model.py \
  --model-type transformer \
  --batch-size 128 \
  --hidden-size 256 \
  --amp \
  --num-workers 8
```

### 2. 多GPU训练
```bash
# 使用DataParallel
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_universal_model.py \
  --batch-size 512 \
  --amp
```

### 3. 梯度累积（模拟更大batch）
```bash
python scripts/train_universal_model.py \
  --batch-size 128 \
  --gradient-accumulation-steps 4 \
  --amp
# 等效于 batch_size=512
```

## 📝 总结

### 立即执行的优化
1. ✅ 添加 `--amp` 参数
2. ✅ 增大 `--batch-size` 到 256
3. ✅ 增大 `--hidden-size` 到 256
4. ✅ 增加 `--num-workers` 到 8
5. ✅ 确保 `--pin-memory` 启用

### 预期结果
- 🚀 训练速度提升 **5-10倍**
- 💪 GPU使用率提升到 **60-90%**
- 📈 显存使用提升到 **8-12GB**
- ⚡ 每轮训练时间从 **5分钟** 降到 **30-60秒**

### 下一步
```bash
# 1. 运行诊断
python diagnose_gpu_usage.py

# 2. 使用优化脚本
train_gpu_optimized.bat  # Windows
# 或
bash train_gpu_optimized.sh  # Linux/Mac

# 3. 监控GPU
nvidia-smi -l 1
```

---

**需要帮助？** 运行 `python diagnose_gpu_usage.py` 获取详细诊断报告。
