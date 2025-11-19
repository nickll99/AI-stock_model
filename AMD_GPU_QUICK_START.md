# AMD GPU 快速开始

## ⚡ 一键命令（推荐）

```bash
# AMD GPU优化配置
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 100 \
    --batch-size 256 \
    --hidden-size 256 \
    --device cuda \
    --num-workers 0 \
    --limit 1000
```

**关键点：**
- ✅ `--device cuda` - ROCm兼容CUDA API
- ✅ `--num-workers 0` - 避免内存爆炸
- ✅ `--limit 1000` - 限制股票数量，减少内存占用

---

## 🔍 第1步：诊断环境

```bash
python diagnose_gpu.py
```

**检查项：**
- ✅ PyTorch支持ROCm
- ✅ GPU可用（`torch.cuda.is_available() == True`）
- ✅ 显存充足

---

## 🚀 第2步：快速测试（5分钟）

```bash
python scripts/train_universal_model.py \
    --device cuda \
    --limit 100 \
    --epochs 5 \
    --num-workers 0
```

**预期输出：**
```
配置:
  设备: cuda          ← 确认使用GPU
  ...

创建模型...
✓ 模型参数量: 2,345,678

开始训练...
Epoch [1/5] - ... Time: 12.34s  ← 每个epoch 10-30秒
```

---

## 📊 第3步：监控GPU

```bash
# 在另一个终端运行
watch -n 1 rocm-smi
```

**目标状态：**
- GPU利用率：80-100% ✅
- 显存使用：70-90% ✅
- 温度：70-85°C ✅

---

## 💡 内存优化

### 问题：内存占用50%

**解决方案：**

```bash
# 1. 限制股票数量
--limit 500

# 2. 减小batch size
--batch-size 128

# 3. 禁用workers
--num-workers 0

# 4. 使用更小的模型
--hidden-size 128
```

### 完整命令

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

---

## 🎯 根据显存选择配置

### 8GB显存

```bash
python scripts/train_universal_model.py \
    --batch-size 128 \
    --hidden-size 128 \
    --device cuda \
    --num-workers 0 \
    --limit 500
```

### 12GB显存

```bash
python scripts/train_universal_model.py \
    --batch-size 256 \
    --hidden-size 128 \
    --device cuda \
    --num-workers 0 \
    --limit 1000
```

### 16GB+显存

```bash
python scripts/train_universal_model.py \
    --batch-size 256 \
    --hidden-size 256 \
    --device cuda \
    --num-workers 0
```

---

## 🔧 常见问题

### Q1: GPU不可用？

```bash
# 检查
python -c "import torch; print(torch.cuda.is_available())"

# 如果输出False，重新安装PyTorch ROCm版本
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6
```

### Q2: 内存占用太高？

```bash
# 使用内存优化配置
python scripts/train_universal_model.py \
    --device cuda \
    --batch-size 128 \
    --num-workers 0 \
    --limit 500
```

### Q3: GPU利用率低？

**检查清单：**
1. ✅ 是否使用了 `--device cuda`？
2. ✅ 是否使用了缓存数据？
3. ✅ Batch size是否足够大？

**解决方案：**
```bash
# 增大batch size
--batch-size 512

# 确保使用缓存
python scripts/prepare_training_data.py --symbols all --workers 8 --resume
```

---

## 📚 详细文档

- **docs/AMD_GPU训练配置指南.md** - AMD GPU完整指南
- **TRAINING_CONFIGS.md** - 所有训练配置详解
- **diagnose_gpu.py** - GPU诊断工具

---

## ✨ 完整工作流

```bash
# 1. 诊断环境
python diagnose_gpu.py

# 2. 确保数据已预热
python scripts/prepare_training_data.py --symbols all --workers 8 --resume

# 3. 快速测试
python scripts/train_universal_model.py \
    --device cuda \
    --limit 100 \
    --epochs 5 \
    --num-workers 0

# 4. 监控GPU（另一个终端）
watch -n 1 rocm-smi

# 5. 正式训练
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 100 \
    --batch-size 256 \
    --hidden-size 256 \
    --device cuda \
    --num-workers 0 \
    --limit 1000
```

---

## 🎯 关键要点

1. ✅ **使用 `--device cuda`** - ROCm兼容CUDA API
2. ✅ **使用 `--num-workers 0`** - AMD GPU必须设置
3. ✅ **限制股票数量** - 从小规模开始（`--limit 500`）
4. ✅ **监控GPU** - 使用 `rocm-smi`
5. ✅ **使用缓存数据** - 避免从数据库加载

现在你可以在AMD GPU上高效训练了！🚀
