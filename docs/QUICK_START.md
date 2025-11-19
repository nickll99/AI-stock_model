# 快速开始指南

## 系统要求

- Python 3.10+
- MySQL 8.0+
- Redis 6.0+
- MinIO (可选，用于模型存储)

## 安装步骤

### 1. 克隆项目

```bash
git clone <repository-url>
cd AI-stock_model
```

### 2. 创建虚拟环境

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

**主要依赖：**
- FastAPI - Web框架
- PyTorch - 深度学习框架
- SQLAlchemy - ORM
- Pandas/NumPy - 数据处理
- Redis - 缓存
- Celery - 异步任务

### 4. 配置环境变量

复制环境变量模板：

```bash
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac
```

编辑 `.env` 文件：

```env
# 数据库配置
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=stock_db

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# MinIO配置
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=ai-models
MINIO_SECURE=false

# Celery配置
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# 应用配置
APP_ENV=development
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
LOG_JSON_FORMAT=true
```

### 5. 准备数据库

**创建数据库：**

```sql
CREATE DATABASE stock_db CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
```

**初始化表结构：**

```bash
python src/database/init_db.py
```

**导入股票数据：**

确保 `stock_basic_info` 和 `stock_kline_data` 表中有数据。

### 6. 启动服务

**启动Redis：**

```bash
# Windows
redis-server

# Linux/Mac
sudo service redis-server start
```

**启动MinIO（可选）：**

```bash
# Windows
minio.exe server E:\minio-data

# Linux/Mac
minio server /data/minio
```

**启动Celery Worker：**

```bash
celery -A src.tasks.celery_app worker --loglevel=info
```

**启动API服务：**

```bash
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## 验证安装

### 1. 快速测试

```bash
python quick_test.py
```

**预期输出：**
```
✓ 所有测试通过！系统准备就绪。
```

### 2. 访问API文档

打开浏览器访问：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- 健康检查: http://localhost:8000/health

### 3. 测试API

```bash
# 健康检查
curl http://localhost:8000/health

# 获取股票列表
curl http://localhost:8000/api/v1/data/stocks?limit=10

# 获取K线数据
curl http://localhost:8000/api/v1/data/stocks/000001/kline?limit=100
```

## 运行测试

### 快速测试（2分钟）

```bash
python quick_test.py
```

测试内容：
- 模块导入
- 数据库连接
- 模型创建
- 特征工程

### 完整测试（10-15分钟）

```bash
python test_training_prediction.py
```

测试内容：
- 数据加载
- 特征工程
- 模型训练
- 模型评估
- 预测功能

## 训练第一个模型

### 方法1：使用测试脚本

```bash
# 编辑 test_training_prediction.py
# 修改 epochs=50（在 test_model_training 函数中）

python test_training_prediction.py
```

### 方法2：使用API

```bash
# 启动训练任务
curl -X POST http://localhost:8000/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "stock_code": "000001",
    "model_type": "lstm",
    "config": {
      "epochs": 50,
      "batch_size": 32,
      "learning_rate": 0.001
    }
  }'

# 查询训练状态
curl http://localhost:8000/api/v1/training/status/{task_id}
```

### 方法3：使用Python脚本

创建 `train_model.py`：

```python
from src.data.loader import StockDataLoader
from src.features.dataset_builder import FeatureDatasetBuilder
from src.models.lstm_model import LSTMModel
from src.training.trainer import ModelTrainer

# 加载数据
loader = StockDataLoader()
df = loader.load_kline_data('000001', '2021-01-01', '2024-12-31')

# 构建数据集
builder = FeatureDatasetBuilder()
df_features = builder.build_feature_matrix(df)
X, y, _ = builder.prepare_sequences(df_features, seq_length=60)
X_train, X_val, X_test, y_train, y_val, y_test = builder.split_dataset(X, y)

# 创建模型
model = LSTMModel(input_size=X_train.shape[2], hidden_size=128, num_layers=2)

# 训练
trainer = ModelTrainer(model)
train_loader = trainer.create_data_loader(X_train, y_train)
val_loader = trainer.create_data_loader(X_val, y_val)

history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    learning_rate=0.001
)

print(f"训练完成！最佳验证损失: {history['best_val_loss']:.6f}")
```

运行：

```bash
python train_model.py
```

## 进行预测

### 方法1：使用API

```bash
curl -X POST http://localhost:8000/api/v1/prediction/predict \
  -H "Content-Type: application/json" \
  -d '{
    "stock_code": "000001",
    "days": 5,
    "model_version": "latest"
  }'
```

### 方法2：使用Python脚本

```python
from src.prediction.engine import PredictionEngine

# 创建预测引擎
engine = PredictionEngine(
    model_path='checkpoints/lstm/best_model.pth',
    model_type='lstm',
    model_config={
        'input_size': 45,
        'seq_length': 60,
        'hidden_size': 128,
        'num_layers': 2
    }
)

# 预测
result = engine.predict('000001', days=5)

print(f"股票: {result['symbol']}")
print(f"趋势: {result['trend']}")
for pred in result['predictions']:
    print(f"{pred['date']}: {pred['price']:.2f}")
```

## 常见问题

### Q: 安装PyTorch失败

**A:** 访问 https://pytorch.org/ 选择适合你系统的安装命令。

对于Windows + CUDA 11.8：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

对于CPU版本：
```bash
pip install torch torchvision torchaudio
```

### Q: 数据库连接失败

**A:** 检查：
1. MySQL服务是否启动
2. .env文件配置是否正确
3. 数据库是否已创建
4. 用户权限是否正确

### Q: Redis连接失败

**A:** 检查：
1. Redis服务是否启动
2. 端口6379是否被占用
3. 防火墙设置

### Q: 内存不足

**A:** 
1. 减小batch_size
2. 减小模型大小
3. 使用更少的数据
4. 增加系统内存

### Q: 训练很慢

**A:**
1. 使用GPU（如果可用）
2. 减小数据量（测试用）
3. 使用更小的模型
4. 减少epoch数

## 下一步

1. **阅读文档**
   - [测试指南](TESTING_GUIDE.md)
   - [API文档](API_DOCUMENTATION.md)
   - [开发指南](DEVELOPMENT_GUIDE.md)

2. **训练模型**
   - 使用完整数据集
   - 调优超参数
   - 尝试不同模型

3. **开发前端**
   - React + TypeScript
   - ECharts可视化
   - 实时数据展示

4. **部署**
   - Docker容器化
   - 生产环境配置
   - 监控和日志

## 获取帮助

- 查看文档：`docs/` 目录
- 查看示例：`examples/` 目录
- 运行测试：`python quick_test.py`
- 查看日志：`logs/app.log`

## 项目结构

```
AI-stock_model/
├── src/                    # 源代码
│   ├── api/               # API服务
│   ├── data/              # 数据加载
│   ├── features/          # 特征工程
│   ├── models/            # 深度学习模型
│   ├── training/          # 训练模块
│   ├── prediction/        # 预测模块
│   ├── database/          # 数据库
│   ├── cache/             # 缓存
│   └── utils/             # 工具
├── docs/                  # 文档
├── examples/              # 示例代码
├── tests/                 # 测试
├── checkpoints/           # 模型检查点
├── logs/                  # 日志
├── requirements.txt       # 依赖
├── .env.example          # 环境变量模板
└── README.md             # 项目说明
```

祝你使用愉快！🚀
