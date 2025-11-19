# A股市场AI模型系统

基于深度学习的A股市场智能分析和预测平台。

## 功能特性

- 📊 股票数据加载和预处理
- 🤖 多种深度学习模型（LSTM、GRU、Transformer）
- 📈 技术指标计算（MA、MACD、RSI、布林带等）
- 🔮 股票价格预测（带置信区间）
- 📱 可视化Web界面
- ⚡ 异步任务处理
- 💾 模型版本管理

## 技术栈

- **后端**: Python 3.10+, FastAPI
- **机器学习**: PyTorch, scikit-learn
- **数据库**: MySQL, Redis
- **对象存储**: MinIO
- **任务队列**: Celery
- **前端**: React, TypeScript, ECharts

## 快速开始

### 1. 环境准备

```bash
# 安装Python依赖
pip install -r requirements.txt

# 复制环境配置文件
cp .env.example .env

# 编辑.env文件，配置数据库连接信息
```

### 2. 数据库初始化

```bash
# 创建AI相关表
python -m src.database.init_db
```

### 3. 启动服务

```bash
# 启动FastAPI服务
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 启动Celery Worker（另一个终端）
celery -A src.celery_app worker --loglevel=info
```

## 项目结构

```
.
├── src/
│   ├── config.py              # 配置管理
│   ├── database/              # 数据库相关
│   │   ├── connection.py      # 数据库连接
│   │   ├── models.py          # ORM模型
│   │   └── init_db.py         # 数据库初始化
│   ├── data/                  # 数据加载模块
│   ├── features/              # 特征工程
│   ├── models/                # 深度学习模型
│   ├── training/              # 模型训练
│   ├── prediction/            # 预测服务
│   └── api/                   # API路由
├── requirements.txt           # Python依赖
├── .env.example              # 环境配置示例
└── README.md                 # 项目文档
```

## 开发状态

🚧 项目正在开发中...

## License

MIT
