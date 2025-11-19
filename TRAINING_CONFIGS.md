# 训练配置详解

## 🎯 快速开始

### 第1步：诊断环境

```bash
python diagnose_gpu.py
```

### 第2步：选择配置

根据诊断结果选择合适的配置。

---

## 📋 所有可用参数

### 模型参数

| 参数 | 默认值 | 说明 | 推荐值 |
|------|--------|------|--------|
| `--model-type` | lstm | 模型类型 (lstm/gru/transformer) | transformer |
| `--hidden-size` | 128 | 隐藏层大小 | 128-256 |
| `--num-layers` | 2 | 网络层数 | 2-4 |
| `--dropout` | 0.2 | Dropout率 | 0.1-0.3 |
| `--stock-embedding-dim` | 32 | 股票嵌入维度 | 32-64 |

### 训练参数

| 参数 | 默认值 | 说明 | 推荐值 |
|------|--------|------|--------|
| `--epochs` | 50 | 训练轮数 | 50-100 |
| `--batch-size` | 128 | 批次大小 | 128-512 |
| `--learning-rate` | 0.001 | 学习率 | 0.0001-0.001 |

### 性能参数

| 参数 | 默认值 | 说明 | 推荐值 |
|------|--------|------|--------|
| `--device` | cuda/cpu | 计算设备 | cuda |
| `--amp` | False | 混合精度训练 | 启用（GPU） |
| `--num-workers` | 4 | DataLoader进程数 | 0（AMD GPU） |
| `--pin-memory` | True | 固定内存 | True（GPU） |
| `--gradient-accumulation-steps` | 1 | 梯度累积步数 | 1-4 |

### 数据参数

| 参数 | 默认值 | 说明 | 推荐值 |
|------|--------|------|--------|
| `--limit` | None | 限制股票数量 | 500-1000（测试） |
| `--stock-type` | None | 股票类型筛选 | None |
| `--no-cache` | False | 不使用缓存 | False |
| `--kline-cache-dir` | data/parquet | K线缓存目录 | data/parquet |
| `--feature-cache-dir` | data/features | 特征缓存目录 | data/features |

### 输出参数

| 参数 | 默认值 | 说明 | 推荐值 |
|------|--------|------|--------|
| `--output-dir` | out/universal_model | 输出目录 | 自定义 |

---

## 🎨 预设配置

### 配置1：快速测试（5分钟）

**适用场景：** 验证环境和代码

```bash
python scripts/train_universal_model.py \
    --model-type lstm \
    --epochs 5 \
    --batch-size 128 \
    --hidden-size 64 \
    --device cuda \
    --num-workers 0 \
    --limit 100
```

**预期结果：**
- 训练时间：5分钟
- 内存占用：2-4GB
- GPU显存：1-2GB

### 配置2：中等规模（30分钟）

**适用场景：** 模型调试和参数调优

```bash
python scripts/train_universal_model.py \
    --model-type lstm \
    --epochs 20 \
    --batch-size 256 \
    --hidden-size 128 \
    --device cuda \
    --num-workers 0 \
    --limit 500
```

**预期结果：**
- 训练时间：30分钟
- 内存占用：8-12GB
- GPU显存：4-6GB

### 配置3：标准训练（2小时）

**适用场景：** 正式训练

```bash
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 50 \
    --batch-size 256 \
    --hidden-size 128 \
    --device cuda \
    --num-workers 0 \
    --limit 1000
```

**预期结果：**
- 训练时间：2小时
- 内存占用：16-24GB
- GPU显存：8-12GB

### 配置4：全量训练（4-6小时）

**适用场景：** 生产环境

```bash
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 100 \
    --batch-size 256 \
    --hidden-size 256 \
    --device cuda \
    --num-workers 0
```

**预期结果：**
- 训练时间：4-6小时
- 内存占用：32-48GB
- GPU显存：12-16GB

### 配置5：内存受限（推荐AMD GPU）

**适用场景：** 内存或显存不足

```bash
python scripts/train_universal_model.py \
    --model-type lstm \
    --epochs 50 \
    --batch-size 128 \
    --hidden-size 128 \
    --device cuda \
    --num-workers 0 \
    --limit 1000
```

**预期结果：**
- 训练时间：2-3小时
- 内存占用：12-16GB
- GPU显存：4-6GB

### 配置6：高性能（NVIDIA GPU）

**适用场景：** NVIDIA GPU + 充足资源

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

**预期结果：**
- 训练时间：1-2小时
- 内存占用：16-24GB
- GPU显存：12-16GB

---

## 💡 参数调优指南

### Batch Size选择

| 显存大小 | 推荐Batch Size | 说明 |
|---------|---------------|------|
| 4GB | 64-128 | 小模型 |
| 8GB | 128-256 | 标准配置 |
| 12GB | 256-512 | 高性能 |
| 16GB+ | 512-1024 | 极致性能 |

**规则：**
- 显存不足：减小batch size
- GPU利用率低：增大batch size
- 内存不足：减小batch size或限制股票数量

### Hidden Size选择

| 模型复杂度 | Hidden Size | 参数量 | 说明 |
|-----------|-------------|--------|------|
| 小 | 64 | ~500K | 快速训练 |
| 中 | 128 | ~2M | 标准配置 |
| 大 | 256 | ~8M | 高精度 |
| 超大 | 512 | ~32M | 研究用 |

**规则：**
- 数据量大：使用更大的hidden size
- 显存不足：使用更小的hidden size
- 过拟合：减小hidden size或增加dropout

### Epochs选择

| 场景 | Epochs | 说明 |
|------|--------|------|
| 快速测试 | 5-10 | 验证代码 |
| 调试 | 20-30 | 参数调优 |
| 标准训练 | 50-100 | 正式训练 |
| 精细训练 | 100-200 | 追求极致 |

**规则：**
- 使用早停（early stopping）避免过拟合
- 观察验证损失，停止时机合适即可

### Learning Rate选择

| 模型类型 | Learning Rate | 说明 |
|---------|--------------|------|
| LSTM/GRU | 0.001 | 标准 |
| Transformer | 0.0001-0.0005 | 较小 |
| 大模型 | 0.0001 | 更小 |

**规则：**
- 损失不下降：减小learning rate
- 训练不稳定：减小learning rate
- 收敛太慢：增大learning rate

---

## 🔧 内存优化策略

### 问题：内存占用50%

**原因：** 数据加载阶段一次性加载所有股票

**解决方案：**

#### 1. 限制股票数量

```bash
# 从500只开始
--limit 500

# 逐步增加
--limit 1000
--limit 2000
```

#### 2. 减小Batch Size

```bash
# 从128开始
--batch-size 128

# 如果还不够
--batch-size 64
```

#### 3. 禁用DataLoader Workers

```bash
# 避免多进程内存复制
--num-workers 0
```

#### 4. 使用更小的模型

```bash
--model-type lstm \
--hidden-size 64
```

#### 5. 分批训练

```bash
# 第一批
python scripts/train_universal_model.py \
    --limit 1000 \
    --output-dir out/batch1

# 第二批
python scripts/train_universal_model.py \
    --limit 2000 \
    --output-dir out/batch2
```

---

## 📊 性能监控

### GPU监控

```bash
# NVIDIA GPU
watch -n 1 nvidia-smi

# AMD GPU
watch -n 1 rocm-smi
```

### 内存监控

```bash
# 系统内存
watch -n 1 free -h

# 进程内存
watch -n 1 "ps aux | grep train_universal_model | grep -v grep"
```

### 训练日志

关注以下指标：
- 每个epoch时间（应该稳定）
- 训练损失（应该下降）
- 验证损失（应该下降）
- GPU利用率（应该>80%）

---

## 🎯 推荐工作流

### 第1步：环境诊断

```bash
python diagnose_gpu.py
```

### 第2步：快速测试

```bash
python scripts/train_universal_model.py \
    --device cuda \
    --limit 100 \
    --epochs 5 \
    --num-workers 0
```

### 第3步：中等规模

```bash
python scripts/train_universal_model.py \
    --device cuda \
    --limit 500 \
    --epochs 20 \
    --batch-size 256 \
    --num-workers 0
```

### 第4步：全量训练

```bash
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 100 \
    --batch-size 256 \
    --hidden-size 256 \
    --device cuda \
    --num-workers 0
```

---

## ✨ 总结

### AMD GPU推荐配置

```bash
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 100 \
    --batch-size 256 \
    --hidden-size 256 \
    --device cuda \
    --num-workers 0 \
    --limit 1000
```

### NVIDIA GPU推荐配置

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

### 内存受限配置

```bash
python scripts/train_universal_model.py \
    --model-type lstm \
    --epochs 50 \
    --batch-size 128 \
    --hidden-size 128 \
    --device cuda \
    --num-workers 0 \
    --limit 500
```

现在你有完整的训练配置指南了！🚀
