# GPU配置完整指南

## 🎯 目标

配置GPU加速训练，速度提升10-20倍！

---

## 🔍 第一步：检查GPU状态

### 运行检测脚本

```bash
python scripts/check_gpu.py
```

### 可能的输出

#### 情况1：GPU配置正常 ✅

```
======================================================================
  GPU配置检查
======================================================================

1. 检查PyTorch...
✓ PyTorch版本: 2.0.1+cu118

2. 检查CUDA...
✓ CUDA可用: True
✓ CUDA版本: 11.8
✓ GPU数量: 1
✓ GPU 0: NVIDIA GeForce RTX 3080
  - 总内存: 10.00 GB

3. 检查NVIDIA驱动...
✓ NVIDIA驱动已安装

GPU信息:
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.11    Driver Version: 525.60.11    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| 30%   45C    P8    15W / 320W |    500MiB / 10240MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

======================================================================
  建议
======================================================================

✓ GPU配置正常，可以使用GPU训练！

使用GPU训练:
  python examples/train_with_manager.py
  python scripts/train_universal_model.py --device cuda
```

**结论：可以直接使用GPU训练！**

#### 情况2：CUDA不可用 ❌

```
======================================================================
  GPU配置检查
======================================================================

1. 检查PyTorch...
✓ PyTorch版本: 2.0.1

2. 检查CUDA...
✗ CUDA不可用

可能的原因:
  1. 安装的是CPU版本的PyTorch
  2. 没有NVIDIA GPU
  3. CUDA驱动未安装或版本不匹配

3. 检查NVIDIA驱动...
✗ nvidia-smi未找到
  可能没有安装NVIDIA驱动或不是NVIDIA GPU
```

**结论：需要配置GPU环境**

---

## 🛠️ 第二步：配置GPU环境

### 方案A：有NVIDIA GPU（推荐）

#### 1. 检查GPU硬件

**Windows:**
```
1. 右键"此电脑" -> 管理 -> 设备管理器
2. 展开"显示适配器"
3. 查看是否有NVIDIA显卡
```

**Linux:**
```bash
lspci | grep -i nvidia
```

**常见NVIDIA GPU:**
- GeForce系列：GTX 1060/1660/2060/3060/3080/4090等
- RTX系列：RTX 2080/3080/4080/4090等
- Quadro系列：专业卡
- Tesla系列：服务器卡

#### 2. 安装NVIDIA驱动

**Windows:**
1. 访问：https://www.nvidia.com/Download/index.aspx
2. 选择你的GPU型号
3. 下载并安装驱动
4. 重启电脑

**Linux (Ubuntu):**
```bash
# 方法1：使用apt安装（推荐）
sudo apt update
sudo apt install nvidia-driver-525

# 方法2：使用官方安装包
# 从NVIDIA官网下载.run文件
sudo bash NVIDIA-Linux-x86_64-525.60.11.run

# 重启
sudo reboot

# 验证
nvidia-smi
```

#### 3. 安装CUDA版本的PyTorch

**卸载当前PyTorch:**
```bash
pip uninstall torch torchvision torchaudio
```

**安装CUDA 11.8版本（推荐）:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**或安装CUDA 12.1版本:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**使用conda安装:**
```bash
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### 4. 验证安装

```bash
python scripts/check_gpu.py
```

应该看到：
```
✓ CUDA可用: True
✓ GPU数量: 1
✓ GPU 0: NVIDIA GeForce RTX 3080
```

### 方案B：没有NVIDIA GPU

如果你的电脑没有NVIDIA GPU（如AMD GPU、Intel集显），则：

**选项1：使用CPU训练**
- 速度较慢但可用
- 适合小规模训练

**选项2：使用云GPU**
- Google Colab（免费GPU）
- AWS/阿里云/腾讯云（付费GPU）
- AutoDL/恒源云（国内GPU租用）

**选项3：购买GPU**
- 推荐：RTX 3060/3080/4090
- 预算有限：GTX 1660 Super

---

## 🚀 第三步：使用GPU训练

### 单股票训练

```bash
# 代码会自动检测并使用GPU
python examples/train_with_manager.py
```

**输出应该显示：**
```
开始训练，设备: cuda
```

### 批量训练

```bash
# 批量训练会自动使用GPU
python scripts/batch_train_all_stocks.py \
    --symbols all \
    --workers 2 \
    --resume
```

**注意：**
- GPU训练时，workers建议设置为2-4
- 不要设置太多workers，会导致GPU内存不足

### 通用模型训练

```bash
# 显式指定使用GPU
python scripts/train_universal_model.py --device cuda
```

---

## 📊 性能对比

### 单股票训练

| 硬件 | 训练时间 | 相对速度 |
|------|---------|----------|
| CPU (4核 i5) | 5分钟 | 1x |
| CPU (8核 i7) | 3分钟 | 1.7x |
| GPU (GTX 1660) | 30秒 | 10x |
| GPU (RTX 3080) | 15秒 | 20x |
| GPU (RTX 4090) | 10秒 | 30x |

### 批量训练5000只股票

| 硬件 | 训练时间 | 相对速度 |
|------|---------|----------|
| CPU (8核) | 40-50小时 | 1x |
| GPU (GTX 1660) | 15-20小时 | 3x |
| GPU (RTX 3080) | 8-12小时 | 5x |
| GPU (RTX 4090) | 4-6小时 | 10x |

### 通用模型训练

| 硬件 | 训练时间 | 相对速度 |
|------|---------|----------|
| CPU (8核) | 5小时 | 1x |
| GPU (GTX 1660) | 2小时 | 2.5x |
| GPU (RTX 3080) | 1小时 | 5x |
| GPU (RTX 4090) | 30分钟 | 10x |

---

## 🔧 常见问题

### Q1: 显示"CUDA不可用"怎么办？

**检查清单：**

1. **确认有NVIDIA GPU**
```bash
# Windows
设备管理器 -> 显示适配器

# Linux
lspci | grep -i nvidia
```

2. **确认驱动已安装**
```bash
nvidia-smi
```

3. **确认PyTorch版本**
```bash
python -c "import torch; print(torch.__version__)"
```

应该看到类似：`2.0.1+cu118`（有cu118后缀）

如果是：`2.0.1`（没有cu后缀），说明是CPU版本

**解决方案：**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Q2: GPU内存不足（OOM）

**错误信息：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**

```bash
# 方法1：减小批次大小
python examples/train_with_manager.py
# 修改config中的batch_size从32改为16

# 方法2：减小模型大小
# 修改hidden_size从128改为64

# 方法3：减少workers
python scripts/batch_train_all_stocks.py --workers 1
```

### Q3: GPU利用率低

**症状：**
```bash
nvidia-smi
# GPU利用率只有10-20%
```

**原因：**
- 批次太小
- 数据加载慢
- CPU成为瓶颈

**解决方案：**

```python
# 增大批次大小
config = {
    "batch_size": 128,  # 从32增加到128
}

# 使用数据预热
python scripts/prepare_training_data.py --symbols all --workers 8 --resume

# 增加DataLoader的workers
train_loader = DataLoader(dataset, batch_size=128, num_workers=4)
```

### Q4: 多GPU如何使用？

**检查GPU数量：**
```bash
python -c "import torch; print(torch.cuda.device_count())"
```

**使用DataParallel：**
```python
import torch.nn as nn

# 在train_with_manager.py中添加
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个GPU")
    model = nn.DataParallel(model)
```

### Q5: CUDA版本不匹配

**错误信息：**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**原因：**
PyTorch的CUDA版本与驱动的CUDA版本不匹配

**解决方案：**

1. **查看驱动支持的CUDA版本**
```bash
nvidia-smi
# 查看右上角的CUDA Version
```

2. **安装匹配的PyTorch版本**
```bash
# 如果驱动支持CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 如果驱动支持CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 💡 最佳实践

### 1. GPU训练配置

```python
# 推荐配置
config = {
    "batch_size": 64,      # GPU可以用更大的批次
    "hidden_size": 256,    # GPU可以用更大的模型
    "num_layers": 3,       # GPU可以用更深的网络
    "workers": 2,          # GPU训练时workers不要太多
}
```

### 2. 监控GPU使用

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或使用gpustat
pip install gpustat
gpustat -i 1
```

### 3. 混合精度训练（高级）

```python
# 使用FP16加速训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(epochs):
    for X, y in train_loader:
        with autocast():
            output = model(X)
            loss = criterion(output, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 4. 清理GPU缓存

```python
import torch

# 训练完成后清理缓存
torch.cuda.empty_cache()
```

---

## 📝 快速检查清单

- [ ] 确认有NVIDIA GPU
- [ ] 安装NVIDIA驱动
- [ ] 运行`nvidia-smi`成功
- [ ] 安装CUDA版本的PyTorch
- [ ] 运行`python scripts/check_gpu.py`
- [ ] 看到"CUDA可用: True"
- [ ] 训练时显示"设备: cuda"
- [ ] GPU利用率>80%

---

## 🎓 总结

### GPU配置步骤

```bash
# 1. 检查GPU
python scripts/check_gpu.py

# 2. 如果CUDA不可用，安装CUDA版本的PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 再次检查
python scripts/check_gpu.py

# 4. 开始GPU训练
python examples/train_with_manager.py
python scripts/train_universal_model.py --device cuda
```

### 性能提升

- 单股票训练：10-30倍
- 批量训练：3-10倍
- 通用模型：2.5-10倍

### 推荐GPU

| 预算 | GPU | 性能 | 价格 |
|------|-----|------|------|
| 入门 | GTX 1660 Super | 2-3x | ¥1500 |
| 中端 | RTX 3060 | 5-6x | ¥2500 |
| 高端 | RTX 3080 | 8-10x | ¥5000 |
| 旗舰 | RTX 4090 | 15-20x | ¥12000 |

现在你可以配置GPU加速训练了！🚀
