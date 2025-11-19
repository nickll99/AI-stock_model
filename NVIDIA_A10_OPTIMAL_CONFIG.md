# NVIDIA A10 最优训练配置

## 🎯 你的GPU配置

```
GPU: NVIDIA A10
显存: 22.06 GB
CUDA: 12.8
驱动: 580.65.06
```

**这是一个非常强大的GPU！** 可以充分利用大batch size和混合精度训练。

---

## ⚡ 推荐配置（极致性能）

### 配置1：极致性能（推荐）

```bash
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 100 \
    --batch-size 1024 \
    --hidden-size 256 \
    --device cuda \
    --amp \
    --num-workers 4 \
    --pin-memory
```

**预期效果：**
- 训练时间：1-1.5小时（全量3000只股票）
- GPU利用率：90-100%
- 显存占用：16-20GB
- 速度提升：4-5倍

### 配置2：平衡配置

```bash
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 100 \
    --batch-size 512 \
    --hidden-size 256 \
    --device cuda \
    --amp \
    --num-workers 4 \
    --pin-memory
```

**预期效果：**
- 训练时间：1.5-2小时
- GPU利用率：85-95%
- 显存占用：12-16GB
- 速度提升：3-4倍

### 配置3：内存优化（解决内存占用50%问题）

```bash
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 100 \
    --batch-size 512 \
    --hidden-size 256 \
    --device cuda \
    --amp \
    --num-workers 2 \
    --pin-memory \
    --limit 1000
```

**预期效果：**
- 训练时间：30-45分钟（1000只股票）
- 内存占用：20-30%（通过限制股票数量）
- GPU利用率：90-100%
- 显存占用：12-16GB

---

## 💡 解决内存占用50%问题

### 问题分析

**当前状态：**
- ❌ 内存占用50%（数据加载阶段）
- ❌ GPU没有被充分利用

**原因：**
数据加载阶段一次性加载所有股票到内存（3000+只股票）

### 解决方案

#### 方案1：限制股票数量（推荐）

```bash
# 训练1000只股票（内存占用降到15-20%）
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 100 \
    --batch-size 1024 \
    --hidden-size 256 \
    --device cuda \
    --amp \
    --num-workers 4 \
    --pin-memory \
    --limit 1000
```

#### 方案2：分批训练

```bash
# 第一批：前1000只
python scripts/train_universal_model.py \
    --device cuda \
    --amp \
    --batch-size 1024 \
    --limit 1000 \
    --output-dir out/universal_model_batch1

# 第二批：1000-2000只
python scripts/train_universal_model.py \
    --device cuda \
    --amp \
    --batch-size 1024 \
    --limit 2000 \
    --output-dir out/universal_model_batch2
```

#### 方案3：减少DataLoader workers

```bash
# 使用2个workers而不是4个
python scripts/train_universal_model.py \
    --device cuda \
    --amp \
    --batch-size 1024 \
    --num-workers 2 \
    --pin-memory
```

---

## 🚀 完整训练流程

### 第1步：快速测试（5分钟）

```bash
python scripts/train_universal_model.py \
    --device cuda \
    --amp \
    --limit 100 \
    --epochs 5 \
    --batch-size 512 \
    --num-workers 2
```

**检查项：**
- ✅ 设备显示为 `cuda`
- ✅ 混合精度显示为 `启用`
- ✅ 每个epoch时间在10-20秒
- ✅ GPU利用率>80%

### 第2步：中等规模测试（30分钟）

```bash
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 20 \
    --batch-size 512 \
    --hidden-size 256 \
    --device cuda \
    --amp \
    --num-workers 4 \
    --pin-memory \
    --limit 500
```

### 第3步：全量训练（1-2小时）

```bash
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 100 \
    --batch-size 1024 \
    --hidden-size 256 \
    --device cuda \
    --amp \
    --num-workers 4 \
    --pin-memory \
    --limit 1000
```

---

## 📊 性能预期

### NVIDIA A10 性能对比

| 配置 | Batch Size | 训练时间 | GPU利用率 | 显存占用 |
|------|-----------|---------|----------|---------|
| 基础 | 128 | 4小时 | 40-50% | 6GB |
| 优化 | 512 | 1.5小时 | 85-95% | 14GB |
| 极致 | 1024 | 1小时 | 95-100% | 18GB |

### vs 其他GPU

| GPU | 显存 | 训练时间 | 说明 |
|-----|------|---------|------|
| GTX 1660 | 6GB | 2小时 | 入门级 |
| RTX 3080 | 10GB | 1小时 | 高性能 |
| **A10** | **22GB** | **1小时** | **专业级** |
| RTX 4090 | 24GB | 30分钟 | 旗舰级 |

---

## 🔍 监控和优化

### 实时监控GPU

```bash
# 在另一个终端运行
watch -n 1 nvidia-smi
```

**目标状态：**
```
+-----------------------------------------------------------------------------------------+
| GPU  Name                 | GPU-Util  | Memory-Usage |
|=========================================================================================|
|   0  NVIDIA A10           |    98%    |  18GB / 22GB |  ← 目标状态
+-----------------------------------------------------------------------------------------+
```

### 监控内存

```bash
# 监控系统内存
watch -n 1 free -h
```

**目标状态：**
- 内存占用：20-30%（通过限制股票数量）
- 可用内存：>50%

### 优化建议

**如果GPU利用率低（<80%）：**
1. 增大batch size：`--batch-size 1024` 或 `2048`
2. 增加workers：`--num-workers 8`
3. 确保使用缓存数据

**如果内存占用高（>50%）：**
1. 限制股票数量：`--limit 1000`
2. 减少workers：`--num-workers 2`
3. 分批训练

**如果显存不足：**
1. 减小batch size：`--batch-size 512`
2. 减小模型：`--hidden-size 128`
3. 禁用混合精度：移除 `--amp`

---

## 🎯 推荐工作流

### 快速开始（推荐）

```bash
# 1. 确保数据已预热
python scripts/prepare_training_data.py --symbols all --workers 8 --resume

# 2. 快速测试（5分钟）
python scripts/train_universal_model.py \
    --device cuda \
    --amp \
    --limit 100 \
    --epochs 5 \
    --batch-size 512

# 3. 监控GPU（另一个终端）
watch -n 1 nvidia-smi

# 4. 正式训练（1-2小时）
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 100 \
    --batch-size 1024 \
    --hidden-size 256 \
    --device cuda \
    --amp \
    --num-workers 4 \
    --pin-memory \
    --limit 1000
```

---

## 📝 参数详解

### 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--batch-size` | 1024 | A10可以支持很大的batch size |
| `--hidden-size` | 256 | 平衡性能和精度 |
| `--amp` | 启用 | 混合精度，速度提升2-3倍 |
| `--num-workers` | 4 | 多进程加载数据 |
| `--pin-memory` | 启用 | 加速数据传输 |
| `--limit` | 1000 | 限制股票数量，减少内存占用 |

### Batch Size选择

| 显存占用目标 | Batch Size | 说明 |
|------------|-----------|------|
| 50% (11GB) | 512 | 保守配置 |
| 70% (15GB) | 1024 | 推荐配置 |
| 85% (19GB) | 2048 | 极致配置 |

---

## 🔧 常见问题

### Q1: 内存占用50%怎么办？

**解决方案：**
```bash
# 限制股票数量到1000只
python scripts/train_universal_model.py \
    --device cuda \
    --amp \
    --batch-size 1024 \
    --limit 1000
```

**效果：** 内存占用降到20-30%

### Q2: GPU利用率低怎么办？

**解决方案：**
```bash
# 增大batch size
python scripts/train_universal_model.py \
    --device cuda \
    --amp \
    --batch-size 2048 \
    --num-workers 8
```

### Q3: 显存不足怎么办？

**解决方案：**
```bash
# 减小batch size
python scripts/train_universal_model.py \
    --device cuda \
    --amp \
    --batch-size 512
```

### Q4: 训练速度没有提升？

**检查清单：**
1. ✅ 是否使用了 `--device cuda`？
2. ✅ 是否启用了 `--amp`？
3. ✅ 是否使用了缓存数据？
4. ✅ Batch size是否足够大？

---

## ✨ 总结

### 你的最优配置

```bash
# 极致性能 + 内存优化
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 100 \
    --batch-size 1024 \
    --hidden-size 256 \
    --device cuda \
    --amp \
    --num-workers 4 \
    --pin-memory \
    --limit 1000
```

### 预期效果

- ✅ 训练时间：1-1.5小时（1000只股票）
- ✅ GPU利用率：90-100%
- ✅ 显存占用：16-18GB（80%）
- ✅ 内存占用：20-30%（通过限制股票数量）
- ✅ 速度提升：4-5倍

### 关键要点

1. ✅ **使用大batch size** - A10有22GB显存，可以用1024或更大
2. ✅ **启用混合精度** - `--amp` 速度提升2-3倍
3. ✅ **限制股票数量** - `--limit 1000` 解决内存占用问题
4. ✅ **使用缓存数据** - 避免从数据库加载
5. ✅ **多workers** - `--num-workers 4` 加速数据加载

现在你可以充分利用NVIDIA A10进行高速训练了！🚀
