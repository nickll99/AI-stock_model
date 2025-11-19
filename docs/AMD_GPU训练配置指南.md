# AMD GPU (ROCm) 训练配置指南

## 🎯 重要说明

**你的服务器使用AMD显卡（A卡），需要使用ROCm而不是CUDA！**

---

## 📋 当前问题分析

### 问题现象
- ✅ 内存占用50%（数据加载阶段）
- ❌ GPU没有被使用
- ❌ 训练速度慢

### 根本原因
1. **PyTorch可能没有正确安装ROCm版本**
2. **数据加载阶段占用大量内存**
3. **需要使用 `--device cuda` 参数（ROCm兼容CUDA API）**

---

## 🔧 解决方案

### 第1步：检查PyTorch ROCm支持

```bash
# 检查PyTorch是否支持ROCm
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'设备数量: {torch.cuda.device_count()}'); print(f'当前设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"
```

**预期输出（正确配置）：**
```
PyTorch版本: 2.0.0+rocm5.4.2
CUDA可用: True
设备数量: 1
当前设备: AMD Radeon RX 7900 XTX  # 或其他AMD显卡型号
```

**如果输出 `CUDA可用: False`，需要重新安装PyTorch ROCm版本！**

### 第2步：安装PyTorch ROCm版本（如果需要）

```bash
# 卸载现有PyTorch
pip uninstall torch torchvision torchaudio -y

# 安装ROCm版本的PyTorch
# 根据你的ROCm版本选择（通常是5.4或5.6）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# 或者使用ROCm 5.4
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4
```

### 第3步：验证ROCm环境

```bash
# 检查ROCm版本
rocm-smi

# 或者
rocminfo | grep "Name:"
```

### 第4步：优化训练配置

创建训练配置文件 `train_config.sh`：

```bash
#!/bin/bash

# AMD GPU优化配置
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # 根据你的GPU调整
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

# 训练命令
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 100 \
    --batch-size 256 \
    --hidden-size 256 \
    --device cuda \
    --amp \
    --num-workers 2 \
    --limit 500
```

---

## 🚀 推荐训练配置

### 配置1：内存优化（推荐先用这个）

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

**特点：**
- 小batch size，减少内存占用
- 不使用DataLoader workers（避免内存爆炸）
- 限制500只股票（快速测试）
- 使用LSTM（比Transformer省内存）

### 配置2：平衡配置

```bash
python scripts/train_universal_model.py \
    --model-type lstm \
    --epochs 50 \
    --batch-size 256 \
    --hidden-size 128 \
    --device cuda \
    --amp \
    --num-workers 0 \
    --limit 1000
```

**特点：**
- 中等batch size
- 启用混合精度（如果ROCm支持）
- 不使用workers
- 1000只股票

### 配置3：全量训练

```bash
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 100 \
    --batch-size 256 \
    --hidden-size 256 \
    --device cuda \
    --amp \
    --num-workers 0
```

**特点：**
- 全量股票
- Transformer模型
- 混合精度
- 不使用workers（避免内存问题）

---

## 💡 内存优化技巧

### 问题：数据加载占用50%内存

**原因：** 一次性加载所有股票数据到内存

**解决方案：**

#### 方案1：限制股票数量

```bash
# 先用500只股票测试
python scripts/train_universal_model.py \
    --device cuda \
    --limit 500
```

#### 方案2：分批训练

```bash
# 训练前1000只
python scripts/train_universal_model.py \
    --device cuda \
    --limit 1000 \
    --output-dir out/universal_model_batch1

# 训练第1000-2000只
python scripts/train_universal_model.py \
    --device cuda \
    --limit 2000 \
    --output-dir out/universal_model_batch2
```

#### 方案3：禁用DataLoader workers

```bash
# workers=0 避免多进程内存复制
python scripts/train_universal_model.py \
    --device cuda \
    --num-workers 0
```

#### 方案4：减小batch size

```bash
# 使用更小的batch size
python scripts/train_universal_model.py \
    --device cuda \
    --batch-size 64
```

---

## 🔍 监控和诊断

### 监控AMD GPU使用

```bash
# 实时监控GPU
watch -n 1 rocm-smi

# 或者
watch -n 1 "rocm-smi | grep -A 10 'GPU'"
```

**关键指标：**
- GPU使用率：应该接近100%
- 显存使用：应该占用70-90%
- 温度：应该在70-85°C

### 监控内存使用

```bash
# 监控系统内存
watch -n 1 free -h

# 监控进程内存
watch -n 1 "ps aux | grep train_universal_model"
```

### 诊断脚本

创建 `diagnose_gpu.py`：

```python
"""诊断GPU配置"""
import torch
import sys

print("="*70)
print("  PyTorch GPU 诊断")
print("="*70)

# PyTorch版本
print(f"\nPyTorch版本: {torch.__version__}")

# CUDA/ROCm可用性
cuda_available = torch.cuda.is_available()
print(f"CUDA/ROCm可用: {cuda_available}")

if cuda_available:
    # 设备信息
    device_count = torch.cuda.device_count()
    print(f"GPU数量: {device_count}")
    
    for i in range(device_count):
        print(f"\nGPU {i}:")
        print(f"  名称: {torch.cuda.get_device_name(i)}")
        print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
    # 测试GPU计算
    print("\n测试GPU计算...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✅ GPU计算测试成功！")
    except Exception as e:
        print(f"❌ GPU计算测试失败: {e}")
else:
    print("\n❌ CUDA/ROCm不可用！")
    print("\n可能的原因：")
    print("1. PyTorch没有安装ROCm版本")
    print("2. ROCm驱动未正确安装")
    print("3. 环境变量未正确设置")
    print("\n解决方案：")
    print("pip uninstall torch -y")
    print("pip install torch --index-url https://download.pytorch.org/whl/rocm5.6")
    sys.exit(1)

print("\n" + "="*70)
```

运行诊断：

```bash
python diagnose_gpu.py
```

---

## 📊 AMD GPU vs NVIDIA GPU

### API兼容性

| 功能 | NVIDIA (CUDA) | AMD (ROCm) | 说明 |
|------|--------------|-----------|------|
| 设备参数 | `--device cuda` | `--device cuda` | ✅ 相同 |
| 混合精度 | `--amp` | `--amp` | ⚠️ 部分支持 |
| 显存管理 | 自动 | 自动 | ✅ 相同 |
| 监控工具 | `nvidia-smi` | `rocm-smi` | ❌ 不同 |

### 性能差异

| 指标 | NVIDIA RTX 4090 | AMD RX 7900 XTX | 说明 |
|------|----------------|-----------------|------|
| FP32性能 | 82.6 TFLOPS | 61 TFLOPS | NVIDIA更快 |
| FP16性能 | 165 TFLOPS | 122 TFLOPS | NVIDIA更快 |
| 显存 | 24GB | 24GB | 相同 |
| 软件支持 | 优秀 | 良好 | NVIDIA更成熟 |

---

## 🎯 完整训练流程

### 第1步：环境检查

```bash
# 1. 检查ROCm
rocm-smi

# 2. 检查PyTorch
python diagnose_gpu.py

# 3. 检查缓存数据
ls data/parquet/*.parquet | wc -l
ls data/features/*_features.parquet | wc -l
```

### 第2步：小规模测试

```bash
# 使用100只股票快速测试
python scripts/train_universal_model.py \
    --device cuda \
    --limit 100 \
    --epochs 5 \
    --batch-size 128 \
    --num-workers 0
```

**预期输出：**
```
配置:
  ...
  设备: cuda
  混合精度: 禁用

创建模型...
✓ 模型参数量: 2,345,678

开始训练...
======================================================================

Epoch [1/5] - Train Loss: 0.234567, Val Loss: 0.345678, Time: 12.34s
```

**关键检查：**
- ✅ 设备显示为 `cuda`
- ✅ 每个epoch时间合理（10-30秒）
- ✅ 没有内存错误

### 第3步：中等规模测试

```bash
# 使用500只股票测试
python scripts/train_universal_model.py \
    --device cuda \
    --limit 500 \
    --epochs 20 \
    --batch-size 256 \
    --num-workers 0
```

### 第4步：全量训练

```bash
# 全量训练
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 100 \
    --batch-size 256 \
    --hidden-size 256 \
    --device cuda \
    --num-workers 0
```

---

## 🔧 常见问题

### Q1: CUDA不可用怎么办？

**检查：**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**如果输出 `False`：**
```bash
# 重新安装PyTorch ROCm版本
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

### Q2: 内存占用太高怎么办？

**解决方案：**
```bash
# 1. 限制股票数量
--limit 500

# 2. 减小batch size
--batch-size 64

# 3. 禁用workers
--num-workers 0

# 4. 使用更小的模型
--hidden-size 64
```

### Q3: GPU利用率低怎么办？

**可能原因：**
1. 数据加载太慢（使用缓存数据）
2. Batch size太小
3. 模型太小

**解决方案：**
```bash
# 1. 确保使用缓存
python scripts/prepare_training_data.py --symbols all --workers 8 --resume

# 2. 增大batch size
--batch-size 512

# 3. 使用更大的模型
--model-type transformer --hidden-size 256
```

### Q4: 混合精度不支持怎么办？

**症状：**
```
RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'
```

**解决方案：**
```bash
# 不使用混合精度
python scripts/train_universal_model.py \
    --device cuda \
    --batch-size 256
    # 不要加 --amp
```

---

## 📝 推荐配置总结

### 快速测试（5分钟）

```bash
python scripts/train_universal_model.py \
    --device cuda \
    --limit 100 \
    --epochs 5 \
    --batch-size 128 \
    --num-workers 0
```

### 中等规模（30分钟）

```bash
python scripts/train_universal_model.py \
    --device cuda \
    --limit 500 \
    --epochs 20 \
    --batch-size 256 \
    --num-workers 0
```

### 全量训练（2-3小时）

```bash
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 100 \
    --batch-size 256 \
    --hidden-size 256 \
    --device cuda \
    --num-workers 0
```

### 内存受限（推荐）

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

---

## ✨ 总结

### 关键配置

1. ✅ **使用 `--device cuda`** - ROCm兼容CUDA API
2. ✅ **使用 `--num-workers 0`** - 避免内存爆炸
3. ✅ **限制股票数量** - 从小规模开始测试
4. ✅ **监控GPU使用** - 使用 `rocm-smi`

### 诊断清单

- [ ] PyTorch支持ROCm（`torch.cuda.is_available() == True`）
- [ ] 缓存数据已准备
- [ ] 使用 `--device cuda` 参数
- [ ] 使用 `--num-workers 0` 避免内存问题
- [ ] 从小规模测试开始（`--limit 100`）

### 下一步

```bash
# 1. 诊断GPU
python diagnose_gpu.py

# 2. 快速测试
python scripts/train_universal_model.py --device cuda --limit 100 --epochs 5 --num-workers 0

# 3. 监控GPU
watch -n 1 rocm-smi
```

现在你可以在AMD GPU上高效训练了！🚀
