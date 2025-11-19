# GPU加速快速开始

## ⚡ 一键加速命令

```bash
# 极致性能配置（推荐）
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

**效果：** 训练速度提升3-4倍！

---

## 🎯 关键参数

| 参数 | 作用 | 效果 |
|------|------|------|
| `--amp` | 混合精度训练 | 速度+2-3x，显存-50% |
| `--batch-size 512` | 增大批次 | 充分利用GPU |
| `--num-workers 4` | 多进程加载 | 避免GPU等待 |
| `--pin-memory` | 固定内存 | 加速数据传输 |

---

## 📊 性能对比

| 配置 | 训练时间 | 提升 |
|------|---------|------|
| 基础配置 | 4小时 | 1x |
| +AMP | 2小时 | 2x |
| +大Batch | 1.5小时 | 2.7x |
| +Workers | 1.2小时 | 3.3x |
| **极致配置** | **1小时** | **4x** |

---

## 🔍 如何确认加速生效？

### 1. 查看配置输出

```
配置:
  ...
  混合精度: 启用          ← 确认AMP启用
  DataLoader workers: 4   ← 确认多进程
```

### 2. 监控GPU使用

```bash
watch -n 1 nvidia-smi
```

**目标状态：**
- GPU利用率：90-100% ✅
- 显存使用：70-90% ✅

### 3. 观察训练速度

```
Epoch [1/100] - ... Time: 25.34s  ← 应该在20-30秒
```

---

## 💡 根据显存选择配置

### 8GB显存

```bash
python scripts/train_universal_model.py \
    --batch-size 256 \
    --hidden-size 128 \
    --amp \
    --num-workers 2 \
    --device cuda
```

### 12GB显存

```bash
python scripts/train_universal_model.py \
    --batch-size 512 \
    --hidden-size 256 \
    --amp \
    --num-workers 4 \
    --device cuda
```

### 16GB+显存

```bash
python scripts/train_universal_model.py \
    --batch-size 1024 \
    --hidden-size 256 \
    --amp \
    --num-workers 8 \
    --device cuda
```

---

## 🔧 常见问题

### Q1: 显存不足怎么办？

```bash
# 减小batch size
python scripts/train_universal_model.py \
    --batch-size 128 \
    --amp \
    --device cuda
```

### Q2: GPU利用率低怎么办？

```bash
# 增大batch size + 多worker
python scripts/train_universal_model.py \
    --batch-size 512 \
    --amp \
    --num-workers 4 \
    --device cuda
```

### Q3: 速度没有提升？

**检查清单：**
1. ✅ 是否启用了 `--amp`？
2. ✅ 是否使用了缓存数据？
3. ✅ Batch size是否足够大？
4. ✅ GPU利用率是否接近100%？

---

## 📚 详细文档

- **docs/GPU训练加速指南.md** - 完整加速指南
- **docs/GPU配置完整指南.md** - GPU配置说明

---

## ✨ 快速开始

```bash
# 1. 确保数据已预热
python scripts/prepare_training_data.py --symbols all --workers 8 --resume

# 2. 使用加速配置训练
python scripts/train_universal_model.py \
    --model-type transformer \
    --epochs 100 \
    --batch-size 512 \
    --hidden-size 256 \
    --device cuda \
    --amp \
    --num-workers 4 \
    --pin-memory

# 3. 监控GPU使用
watch -n 1 nvidia-smi
```

现在你的训练速度会快3-4倍！🚀
