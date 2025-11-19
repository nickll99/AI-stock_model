# AI股票预测模型 - 训练指南

## 🎯 快速开始

### ⚠️ 重要：必须先预热数据！

训练前**必须**先预热数据（包括K线和特征），否则训练会非常慢！

### 方案1：通用模型（推荐，2.5-3小时）

```bash
# 1. 检查GPU（可选但推荐）
python scripts/check_gpu.py

# 2. 数据预热（必须！1.5-2小时）
python scripts/prepare_training_data.py \
    --symbols all \
    --stock-type 主板 \
    --workers 8 \
    --resume

# 3. 训练通用模型（1小时）
python scripts/train_universal_model.py \
    --stock-type 主板 \
    --device cuda
```

**为什么必须预热？**
- ✅ 预热后训练速度提升10-60倍
- ✅ 预热一次，可以多次训练
- ❌ 不预热直接训练会非常慢（10+小时）

### 方案2：独立模型（40-50小时）

```bash
# 1. 数据预热
python scripts/prepare_training_data.py --symbols all --workers 8 --resume

# 2. 批量训练
python scripts/batch_train_all_stocks.py --symbols all --workers 4 --resume
```

---

## 📚 完整文档

### 核心文档

1. **[GPU配置完整指南](docs/GPU配置完整指南.md)** ⭐
   - GPU检测和配置
   - CUDA安装
   - 性能对比
   - 故障排除

2. **[通用模型快速开始](docs/通用模型快速开始.md)** ⭐
   - 3步快速开始
   - 一个模型预测所有股票
   - 训练快10倍

3. **[通用模型vs独立模型对比](docs/通用模型vs独立模型对比.md)**
   - 详细对比分析
   - 性能测试
   - 使用场景

4. **[完整训练使用指南](docs/完整训练使用指南.md)**
   - 环境准备
   - 数据预热
   - 模型训练
   - 参数调优

5. **[全市场批量训练指南](docs/全市场批量训练指南.md)**
   - 批量训练流程
   - 并发处理
   - 断点续传

6. **[并发数据预热使用指南](docs/并发数据预热使用指南.md)**
   - 多进程并发
   - 断点续传
   - 性能优化

---

## 🔍 GPU配置（重要）

### 检查GPU状态

```bash
python scripts/check_gpu.py
```

### 如果显示"CUDA不可用"

```bash
# 1. 卸载CPU版本的PyTorch
pip uninstall torch torchvision torchaudio

# 2. 安装CUDA版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 再次检查
python scripts/check_gpu.py
```

### GPU vs CPU性能对比

| 任务 | CPU (8核) | GPU (RTX 3080) | 提升 |
|------|----------|----------------|------|
| 单股票训练 | 3分钟 | 15秒 | 12x |
| 5000只股票 | 40小时 | 8小时 | 5x |
| 通用模型 | 5小时 | 1小时 | 5x |

---

## 📊 方案对比

### 通用模型 vs 独立模型

| 指标 | 通用模型 | 独立模型 |
|------|---------|---------|
| 训练时间 | 2-5小时 | 40-50小时 |
| 存储空间 | 50MB | 50GB |
| 新股票 | 直接预测 | 需要训练 |
| 精度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 维护成本 | 低 | 高 |
| **推荐度** | ✅ 强烈推荐 | ⚠️  特定场景 |

---

## 🚀 使用场景

### 场景1：生产环境（推荐通用模型）

```bash
# 每周重训练一次
python scripts/prepare_training_data.py --symbols all --workers 8 --resume
python scripts/train_universal_model.py --device cuda
```

**优势：**
- 训练快（2-5小时）
- 维护简单
- 新股票无需训练

### 场景2：研究实验（可选独立模型）

```bash
# 针对特定股票深度优化
python examples/train_with_manager.py
```

**优势：**
- 精度最高
- 可针对性调优

### 场景3：混合方案（最佳实践）

```bash
# 1. 训练通用模型（覆盖全市场）
python scripts/train_universal_model.py --device cuda

# 2. 为重点股票训练专属模型
python scripts/batch_train_all_stocks.py --symbols 000001,600519 --workers 2
```

---

## 📁 项目结构

```
AI-stock_model/
├── scripts/
│   ├── check_gpu.py                    # GPU检测脚本 ⭐
│   ├── prepare_training_data.py        # 数据预热（并发+断点续传）
│   ├── train_universal_model.py        # 训练通用模型 ⭐
│   └── batch_train_all_stocks.py       # 批量训练独立模型
│
├── src/
│   ├── models/
│   │   ├── universal_model.py          # 通用模型 ⭐
│   │   ├── lstm_model.py               # LSTM模型
│   │   ├── gru_model.py                # GRU模型
│   │   └── transformer_model.py        # Transformer模型
│   │
│   ├── training/
│   │   ├── trainer.py                  # 训练器
│   │   ├── evaluator.py                # 评估器
│   │   └── training_manager.py         # 训练管理器
│   │
│   ├── features/
│   │   ├── dataset_builder.py          # 数据集构建器
│   │   └── engineer.py                 # 特征工程
│   │
│   └── data/
│       ├── loader.py                   # 数据加载器
│       ├── cached_loader.py            # 缓存加载器
│       └── preprocessor.py             # 数据预处理
│
├── docs/
│   ├── GPU配置完整指南.md              # GPU配置 ⭐
│   ├── 通用模型快速开始.md             # 快速开始 ⭐
│   ├── 通用模型vs独立模型对比.md       # 方案对比
│   ├── 完整训练使用指南.md             # 完整指南
│   ├── 全市场批量训练指南.md           # 批量训练
│   └── 并发数据预热使用指南.md         # 数据预热
│
└── examples/
    └── train_with_manager.py           # 单股票训练示例
```

---

## 🎓 学习路径

### 新手入门

1. 阅读：[GPU配置完整指南](docs/GPU配置完整指南.md)
2. 运行：`python scripts/check_gpu.py`
3. 阅读：[通用模型快速开始](docs/通用模型快速开始.md)
4. 实践：训练通用模型

### 进阶使用

1. 阅读：[通用模型vs独立模型对比](docs/通用模型vs独立模型对比.md)
2. 阅读：[完整训练使用指南](docs/完整训练使用指南.md)
3. 实践：参数调优

### 高级应用

1. 阅读：[全市场批量训练指南](docs/全市场批量训练指南.md)
2. 实践：批量训练
3. 实践：混合方案

---

## 💡 常见问题

### Q1: 为什么显示使用CPU而不是GPU？

**答：**
1. 检查GPU配置：`python scripts/check_gpu.py`
2. 如果CUDA不可用，安装CUDA版本的PyTorch：
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Q2: 通用模型和独立模型哪个好？

**答：**
- **生产环境**：通用模型（训练快、维护简单）
- **研究实验**：独立模型（精度最高）
- **大型项目**：混合方案

### Q3: 训练需要多长时间？

**答：**
| 方案 | CPU | GPU |
|------|-----|-----|
| 通用模型 | 5小时 | 1小时 |
| 5000只独立模型 | 40小时 | 8小时 |

### Q4: 需要多少存储空间？

**答：**
- 通用模型：50MB
- 5000只独立模型：50GB
- 数据缓存：10-20GB

### Q5: 如何预测新股票？

**答：**
- **通用模型**：直接预测，无需训练
- **独立模型**：需要重新训练

---

## 🔗 相关链接

- [PyTorch官网](https://pytorch.org/)
- [CUDA下载](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA驱动下载](https://www.nvidia.com/Download/index.aspx)

---

## 📞 获取帮助

遇到问题？

1. 查看文档：`docs/`目录
2. 运行检测：`python scripts/check_gpu.py`
3. 查看日志：训练输出和日志文件

---

## 🎉 开始训练

### 最简单的方式（推荐）

```bash
# 1. 检查GPU
python scripts/check_gpu.py

# 2. 数据预热
python scripts/prepare_training_data.py --symbols all --workers 8 --resume

# 3. 训练通用模型
python scripts/train_universal_model.py --device cuda
```

**总耗时：4-8小时（GPU）或 7-10小时（CPU）**

现在开始训练你的AI股票预测模型吧！🚀
