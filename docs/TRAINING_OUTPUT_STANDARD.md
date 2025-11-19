# 训练输出标准

## 概述

本系统采用类似文本大模型的行业标准做法，将所有训练输出统一保存到 `out/` 目录中。数据来源为MySQL数据库。

## 目录结构

```
out/
├── {run_name}/                          # 训练运行目录
│   ├── config.json                      # 训练配置
│   ├── metadata.json                    # 元数据
│   ├── model/                           # 最终模型
│   │   ├── pytorch_model.bin            # 模型权重
│   │   ├── config.json                  # 模型配置
│   │   └── training_args.json           # 训练参数
│   ├── checkpoints/                     # 训练检查点
│   │   ├── checkpoint-10/               # 第10轮检查点
│   │   │   ├── pytorch_model.bin
│   │   │   └── metrics.json
│   │   ├── checkpoint-20/
│   │   ├── ...
│   │   └── best/                        # 最佳模型
│   │       ├── pytorch_model.bin
│   │       └── metrics.json
│   ├── logs/                            # 训练日志
│   │   ├── training.log                 # 训练日志
│   │   └── metrics.json                 # 指标记录
│   └── results/                         # 评估结果
│       ├── eval_results.json            # 评估指标
│       └── predictions.json             # 预测结果
```

## 运行命名规范

运行名称格式：`{stock_symbol}_{model_type}_{timestamp}`

**示例：**
- `000001_lstm_20250119_143025` - 平安银行的LSTM模型
- `600519_gru_20250119_150000` - 贵州茅台的GRU模型
- `transformer_20250119_160000` - 通用Transformer模型

## 文件说明

### 1. config.json - 训练配置

包含完整的训练配置信息。

**示例：**
```json
{
  "run_name": "000001_lstm_20250119_143025",
  "stock_symbol": "000001",
  "model_type": "lstm",
  "data_source": "MySQL",
  "created_at": "2025-01-19T14:30:25",
  "seq_length": 60,
  "hidden_size": 128,
  "num_layers": 2,
  "dropout": 0.2,
  "epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.001,
  "patience": 10,
  "train_start_date": "2021-01-01",
  "train_end_date": "2024-12-31"
}
```

### 2. metadata.json - 元数据

记录训练运行的元数据和状态。

**示例：**
```json
{
  "run_name": "000001_lstm_20250119_143025",
  "stock_symbol": "000001",
  "model_type": "lstm",
  "data_source": "MySQL",
  "created_at": "2025-01-19T14:30:25",
  "updated_at": "2025-01-19T16:45:30",
  "status": "completed",
  "model_saved": true,
  "model_path": "out/000001_lstm_20250119_143025/model/pytorch_model.bin",
  "evaluation_completed": true,
  "eval_metrics": {
    "mae": 0.123456,
    "rmse": 0.234567,
    "mape": 5.67,
    "r2": 0.8765,
    "direction_accuracy": 62.34
  }
}
```

**状态说明：**
- `initialized` - 已初始化
- `preparing` - 准备数据
- `training` - 训练中
- `completed` - 已完成
- `failed` - 失败

### 3. model/pytorch_model.bin - 模型权重

PyTorch模型的state_dict，包含所有模型参数。

**加载方法：**
```python
import torch
from src.models.lstm_model import LSTMModel

# 加载配置
with open('out/run_name/model/config.json', 'r') as f:
    config = json.load(f)

# 创建模型
model = LSTMModel(
    input_size=config['input_size'],
    hidden_size=config['hidden_size'],
    num_layers=config['num_layers']
)

# 加载权重
model.load_state_dict(torch.load('out/run_name/model/pytorch_model.bin'))
```

### 4. model/config.json - 模型配置

模型架构配置。

**示例：**
```json
{
  "model_type": "lstm",
  "input_size": 45,
  "hidden_size": 128,
  "num_layers": 2,
  "output_size": 1,
  "dropout": 0.2,
  "feature_names": ["close", "ma_5", "ma_10", ...]
}
```

### 5. model/training_args.json - 训练参数

训练过程的参数和结果。

**示例：**
```json
{
  "epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.001,
  "optimizer": "Adam",
  "loss_function": "MSELoss",
  "train_samples": 5000,
  "val_samples": 1080,
  "best_val_loss": 0.009654,
  "total_epochs_trained": 45
}
```

### 6. checkpoints/ - 训练检查点

定期保存的模型检查点。

**checkpoint-{epoch}/pytorch_model.bin：**
包含完整的训练状态：
```python
{
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'metrics': {'train_loss': 0.01, 'val_loss': 0.012},
    'timestamp': '2025-01-19T15:00:00'
}
```

**best/：**
最佳模型的检查点（验证损失最低）。

### 7. logs/training.log - 训练日志

文本格式的训练日志。

**示例：**
```
[2025-01-19 14:30:25] 开始加载数据
[2025-01-19 14:30:30] 数据加载完成: 5000 条记录
[2025-01-19 14:30:35] 开始特征工程
[2025-01-19 14:30:40] 特征工程完成: (5000, 60, 45)
[2025-01-19 14:30:45] 创建模型
[2025-01-19 14:30:50] 开始训练
[2025-01-19 14:35:00] Epoch 1: train_loss=0.012345, val_loss=0.013456
...
```

### 8. logs/metrics.json - 指标记录

JSON格式的训练指标记录。

**示例：**
```json
[
  {
    "epoch": 1,
    "phase": "train",
    "timestamp": "2025-01-19T14:35:00",
    "train_loss": 0.012345,
    "val_loss": 0.013456
  },
  {
    "epoch": 2,
    "phase": "train",
    "timestamp": "2025-01-19T14:36:00",
    "train_loss": 0.010234,
    "val_loss": 0.011234
  }
]
```

### 9. results/eval_results.json - 评估结果

模型在测试集上的评估结果。

**示例：**
```json
{
  "evaluated_at": "2025-01-19T16:45:00",
  "stock_symbol": "000001",
  "model_type": "lstm",
  "mae": 0.123456,
  "rmse": 0.234567,
  "mape": 5.67,
  "r2": 0.8765,
  "direction_accuracy": 62.34,
  "test_samples": 1080
}
```

### 10. results/predictions.json - 预测结果

模型的预测结果（可选）。

## 使用方法

### 1. 使用TrainingManager进行训练

```python
from src.training.training_manager import TrainingManager

# 初始化管理器
manager = TrainingManager(
    stock_symbol="000001",
    model_type="lstm",
    output_dir="out"
)

# 保存配置
manager.save_config(config)

# 训练过程中...
manager.save_training_log("开始训练")
manager.log_metrics(epoch, metrics)
manager.save_checkpoint(model, optimizer, epoch, metrics)

# 训练完成
manager.save_model(model, model_config, training_args)
manager.save_evaluation_results(eval_results)
manager.update_status("completed")
```

### 2. 运行标准化训练

```bash
python examples/train_with_manager.py
```

### 3. 查看训练运行

```bash
# 列出所有运行
python tools/list_training_runs.py

# 查看特定运行详情
python tools/list_training_runs.py --run 000001_lstm_20250119_143025
```

### 4. 加载已训练的模型

```python
from src.training.training_manager import TrainingManager

# 加载运行
manager = TrainingManager.load_run("000001_lstm_20250119_143025")

# 获取模型路径
model_path = manager.get_model_path()

# 加载模型配置
model_config = manager.load_model_config()

# 获取摘要
summary = manager.get_summary()
```

## 数据来源

所有训练数据来自MySQL数据库：

**数据表：**
- `stock_basic_info` - 股票基本信息
- `stock_kline_data` - 股票日线数据

**数据加载：**
```python
from src.data.loader import StockDataLoader

loader = StockDataLoader()
df = loader.load_kline_data(
    symbol="000001",
    start_date="2021-01-01",
    end_date="2024-12-31"
)
```

## 最佳实践

### 1. 命名规范

- 使用有意义的运行名称
- 包含股票代码和模型类型
- 添加时间戳避免冲突

### 2. 定期保存检查点

```python
# 每10个epoch保存一次
checkpoint_interval = 10
```

### 3. 记录详细日志

```python
manager.save_training_log("重要事件或状态变化")
manager.log_metrics(epoch, metrics)
```

### 4. 保存完整配置

```python
config = {
    "stock_symbol": "000001",
    "model_type": "lstm",
    "data_source": "MySQL",
    # ... 所有训练参数
}
manager.save_config(config)
```

### 5. 评估和验证

```python
# 训练完成后立即评估
metrics = evaluator.evaluate(test_loader)
manager.save_evaluation_results(metrics)
```

## 与其他系统集成

### 1. 模型服务

```python
# 从out目录加载模型用于预测服务
from src.prediction.engine import PredictionEngine

engine = PredictionEngine(
    model_path="out/000001_lstm_20250119_143025/model/pytorch_model.bin",
    model_type="lstm",
    model_config=model_config
)
```

### 2. 模型版本管理

```python
# 列出所有训练运行
runs = TrainingManager.list_runs()

# 按性能排序
runs_sorted = sorted(runs, key=lambda x: x.get('eval_metrics', {}).get('mae', float('inf')))

# 选择最佳模型
best_run = runs_sorted[0]
```

### 3. 实验追踪

所有训练运行的元数据都保存在 `metadata.json` 中，可以用于：
- 实验对比
- 性能分析
- 模型选择
- 超参数调优

## 清理和维护

### 删除旧的训练运行

```python
import shutil
from pathlib import Path

# 删除特定运行
run_dir = Path("out/000001_lstm_20250119_143025")
if run_dir.exists():
    shutil.rmtree(run_dir)
```

### 备份重要模型

```bash
# 备份最佳模型
cp -r out/000001_lstm_20250119_143025 backups/
```

## 相关文档

- [快速开始指南](QUICK_START.md)
- [训练指南](TRAINING_GUIDE.md)
- [配置指南](CONFIGURATION_GUIDE.md)
- [API文档](API_DOCUMENTATION.md)
