# 训练输出标准化总结

## 完成的工作

根据你的要求，我已经实现了按照文本大模型行业标准的训练输出管理系统，所有输出保存到 `out/` 文件夹，数据来源为MySQL。

## 核心组件

### 1. TrainingManager - 训练管理器

**文件：** `src/training/training_manager.py`

**功能：**
- ✓ 标准化目录结构管理
- ✓ 配置和元数据保存
- ✓ 模型和检查点管理
- ✓ 训练日志记录
- ✓ 评估结果保存
- ✓ 运行状态追踪

**目录结构（参考大模型标准）：**
```
out/
├── {run_name}/
│   ├── config.json              # 训练配置
│   ├── metadata.json            # 元数据
│   ├── model/                   # 最终模型
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── training_args.json
│   ├── checkpoints/             # 训练检查点
│   │   ├── checkpoint-10/
│   │   ├── checkpoint-20/
│   │   └── best/
│   ├── logs/                    # 训练日志
│   │   ├── training.log
│   │   └── metrics.json
│   └── results/                 # 评估结果
│       ├── eval_results.json
│       └── predictions.json
```

### 2. 示例脚本

**文件：** `examples/train_with_manager.py`

**功能：**
- 完整的训练流程示例
- 从MySQL加载数据
- 使用TrainingManager管理输出
- 标准化的训练过程

**运行方法：**
```bash
python examples/train_with_manager.py
```

### 3. 查看工具

**文件：** `tools/list_training_runs.py`

**功能：**
- 列出所有训练运行
- 查看运行详情
- 显示评估指标
- 文件列表和大小

**使用方法：**
```bash
# 列出所有运行
python tools/list_training_runs.py

# 查看特定运行
python tools/list_training_runs.py --run 000001_lstm_20250119_143025
```

### 4. 文档

**文件：** `docs/TRAINING_OUTPUT_STANDARD.md`

**内容：**
- 目录结构说明
- 文件格式规范
- 使用方法
- 最佳实践
- 集成指南

## 主要特性

### 1. 标准化输出结构

参考文本大模型（如Hugging Face Transformers）的标准做法：

```
out/
├── run_1/
│   ├── model/              # 类似 Transformers 的模型目录
│   ├── checkpoints/        # 类似 checkpoint-{step}
│   ├── logs/               # 训练日志
│   └── results/            # 评估结果
```

### 2. 完整的元数据管理

每个训练运行都有完整的元数据：

```json
{
  "run_name": "000001_lstm_20250119_143025",
  "stock_symbol": "000001",
  "model_type": "lstm",
  "data_source": "MySQL",
  "status": "completed",
  "created_at": "2025-01-19T14:30:25",
  "eval_metrics": {...}
}
```

### 3. 数据来源：MySQL

所有训练数据从MySQL数据库加载：

```python
from src.data.loader import StockDataLoader

loader = StockDataLoader()
df = loader.load_kline_data("000001", "2021-01-01", "2024-12-31")
```

### 4. 检查点管理

- 定期保存检查点（每N个epoch）
- 自动保存最佳模型
- 包含完整训练状态

### 5. 日志记录

- 文本日志：`training.log`
- JSON指标：`metrics.json`
- 结构化记录

## 使用示例

### 基本使用

```python
from src.training.training_manager import TrainingManager

# 1. 初始化管理器
manager = TrainingManager(
    stock_symbol="000001",
    model_type="lstm"
)

# 2. 保存配置
manager.save_config(config)

# 3. 训练过程
manager.update_status("training")
manager.save_training_log("开始训练")

# 4. 记录指标
manager.log_metrics(epoch, {"train_loss": 0.01, "val_loss": 0.012})

# 5. 保存检查点
manager.save_checkpoint(model, optimizer, epoch, metrics, is_best=True)

# 6. 保存最终模型
manager.save_model(model, model_config, training_args)

# 7. 保存评估结果
manager.save_evaluation_results(eval_results)

# 8. 完成训练
manager.update_status("completed")
```

### 查看训练运行

```python
# 列出所有运行
runs = TrainingManager.list_runs()

# 加载特定运行
manager = TrainingManager.load_run("000001_lstm_20250119_143025")

# 获取摘要
summary = manager.get_summary()
```

## 运行命名规范

格式：`{stock_symbol}_{model_type}_{timestamp}`

**示例：**
- `000001_lstm_20250119_143025` - 平安银行LSTM模型
- `600519_gru_20250119_150000` - 贵州茅台GRU模型
- `transformer_20250119_160000` - 通用Transformer模型

## 文件说明

| 文件 | 说明 | 格式 |
|------|------|------|
| `config.json` | 训练配置 | JSON |
| `metadata.json` | 运行元数据 | JSON |
| `model/pytorch_model.bin` | 模型权重 | PyTorch |
| `model/config.json` | 模型配置 | JSON |
| `model/training_args.json` | 训练参数 | JSON |
| `checkpoints/checkpoint-{N}/` | 检查点 | 目录 |
| `checkpoints/best/` | 最佳模型 | 目录 |
| `logs/training.log` | 训练日志 | 文本 |
| `logs/metrics.json` | 指标记录 | JSON |
| `results/eval_results.json` | 评估结果 | JSON |
| `results/predictions.json` | 预测结果 | JSON |

## Git配置

已更新 `.gitignore`，忽略 `out/` 目录：

```gitignore
# Training Outputs (following LLM industry standard)
out/
```

## 与现有系统集成

### 1. 训练流程

```python
# 使用TrainingManager替代原有的checkpoint_dir
trainer = ModelTrainer(
    model,
    checkpoint_dir=str(manager.run_dir / "checkpoints")
)
```

### 2. 预测服务

```python
# 从out目录加载模型
from src.prediction.engine import PredictionEngine

engine = PredictionEngine(
    model_path="out/000001_lstm_20250119_143025/model/pytorch_model.bin",
    model_type="lstm",
    model_config=model_config
)
```

### 3. 模型管理

```python
# 列出所有模型
runs = TrainingManager.list_runs()

# 选择最佳模型
best_run = min(runs, key=lambda x: x.get('eval_metrics', {}).get('mae', float('inf')))
```

## 优势

### 1. 标准化

- 遵循行业标准（类似Hugging Face）
- 统一的目录结构
- 一致的文件格式

### 2. 可追溯

- 完整的元数据记录
- 详细的训练日志
- 评估结果保存

### 3. 易于管理

- 清晰的目录结构
- 便于查找和加载
- 支持版本管理

### 4. 便于集成

- 标准化接口
- 易于与其他系统集成
- 支持自动化流程

## 最佳实践

### 1. 使用有意义的运行名称

```python
manager = TrainingManager(
    run_name="000001_lstm_v2_20250119",  # 自定义名称
    stock_symbol="000001",
    model_type="lstm"
)
```

### 2. 定期保存检查点

```python
# 每10个epoch保存
checkpoint_interval = 10
```

### 3. 记录详细日志

```python
manager.save_training_log("数据加载完成: 5000条")
manager.save_training_log(f"Epoch {epoch}: loss={loss:.6f}")
```

### 4. 保存完整配置

```python
config = {
    "stock_symbol": "000001",
    "model_type": "lstm",
    "data_source": "MySQL",
    "seq_length": 60,
    "hidden_size": 128,
    # ... 所有参数
}
manager.save_config(config)
```

### 5. 评估后保存结果

```python
metrics = evaluator.evaluate(test_loader)
manager.save_evaluation_results(metrics)
```

## 测试

### 运行示例训练

```bash
python examples/train_with_manager.py
```

### 查看输出

```bash
# 列出运行
python tools/list_training_runs.py

# 查看详情
python tools/list_training_runs.py --run {run_name}

# 查看文件
ls -lh out/{run_name}/
```

## 相关文档

- [训练输出标准](docs/TRAINING_OUTPUT_STANDARD.md) - 详细规范
- [快速开始](docs/QUICK_START.md) - 入门指南
- [配置指南](docs/CONFIGURATION_GUIDE.md) - 配置说明
- [测试指南](docs/TESTING_GUIDE.md) - 测试方法

## 总结

✅ 已实现按照大模型行业标准的训练输出管理  
✅ 所有输出保存到 `out/` 目录  
✅ 数据来源为MySQL数据库  
✅ 提供完整的管理工具和文档  
✅ 易于集成和使用  

系统现在完全符合文本大模型的标准做法，便于模型管理、版本控制和生产部署！
