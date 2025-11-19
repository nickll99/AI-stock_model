# 配置指南

## 概述

本文档说明A股AI模型系统的配置要求和最佳实践。

## 配置文件

### 1. 环境变量配置 (.env)

**创建配置文件：**

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

**必需配置项：**

```env
# 数据库配置（必需）
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=stock_db

# 应用配置
APP_ENV=development
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
LOG_JSON_FORMAT=true
```

**可选配置项：**

```env
# Redis配置（仅API服务需要，训练不需要）
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# MinIO配置（模型存储）
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=ai-models
MINIO_SECURE=false

# Celery配置（异步任务）
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
```

### 2. 配置说明

#### 数据库配置

系统使用MySQL存储股票数据和模型元数据。

**必需表（已存在）：**
- `stock_basic_info` - 股票基本信息
- `stock_kline_data` - 股票日线数据

**AI相关表（需要创建）：**
- `ai_models` - AI模型元数据
- `ai_predictions` - 预测结果
- `technical_indicators` - 技术指标缓存

**创建AI表：**

```bash
python src/database/init_db.py
```

#### Redis配置

**重要说明：**
- ✓ **训练模块不使用Redis** - 日线级别模型训练直接从数据库读取数据
- ✓ **预测服务可使用Redis** - 缓存预测结果和技术指标，提升API响应速度
- ✓ **API服务使用Redis** - Celery任务队列和结果存储

**使用场景：**
1. API预测结果缓存（TTL: 1小时）
2. 技术指标缓存（TTL: 30分钟）
3. Celery异步任务队列

**不使用场景：**
1. ✗ 模型训练过程
2. ✗ 特征工程计算
3. ✗ 数据加载和预处理

#### MinIO配置

用于存储训练好的模型文件。

**可选配置：**
- 如果不使用MinIO，模型将保存在本地 `checkpoints/` 目录
- 生产环境建议使用MinIO进行集中存储和版本管理

## 配置检查

### 运行配置检查脚本

```bash
python check_config.py
```

**检查内容：**
1. ✓ 环境配置文件是否存在
2. ✓ 配置导入是否正常
3. ✓ 数据库连接是否成功
4. ✓ Redis使用是否符合要求
5. ✓ .gitignore文件是否配置
6. ✓ 目录结构是否完整

**预期输出：**

```
============================================================
  A股AI模型系统 - 配置检查
============================================================

============================================================
  检查环境配置文件
============================================================
✓ .env 文件存在
✓ 必需配置项完整

============================================================
  检查配置导入
============================================================
✓ 配置模块导入成功

当前配置:
  数据库主机: localhost
  数据库端口: 3306
  数据库名称: stock_db
  数据库用户: root
  应用环境: development
  日志级别: INFO

✓ 数据库连接URL: mysql+pymysql://root:***@localhost:3306/stock_db

============================================================
  检查数据库连接
============================================================

尝试连接数据库...
✓ 数据库连接成功

检查数据表...
✓ stock_basic_info 表存在，记录数: 5000
✓ stock_kline_data 表存在，记录数: 1500000

============================================================
  检查Redis使用情况
============================================================
✓ 训练模块未使用Redis缓存（符合日线级别训练要求）

检查预测模块Redis使用...
✓ 预测模块可以使用Redis缓存

============================================================
  检查.gitignore文件
============================================================
✓ .gitignore 文件存在
✓ 关键忽略项已配置

============================================================
  检查目录结构
============================================================
✓ src/data
✓ src/features
✓ src/models
✓ src/training
✓ src/prediction
✓ src/database
✓ src/api
✓ docs
✓ examples

检查运行时目录...
  logs 将在运行时自动创建
  checkpoints 将在运行时自动创建

============================================================
  检查总结
============================================================
环境配置文件          : ✓ 通过
配置导入             : ✓ 通过
数据库连接            : ✓ 通过
Redis使用检查         : ✓ 通过
Git忽略文件          : ✓ 通过
目录结构             : ✓ 通过

总计: 6/6 检查通过

✓ 所有检查通过！系统配置正确。
```

## 配置最佳实践

### 1. 数据库配置

**开发环境：**
```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=dev_password
DB_NAME=stock_db_dev
APP_ENV=development
```

**生产环境：**
```env
DB_HOST=prod-db-server
DB_PORT=3306
DB_USER=stock_app
DB_PASSWORD=strong_password_here
DB_NAME=stock_db
APP_ENV=production
LOG_LEVEL=WARNING
```

### 2. 日志配置

**开发环境：**
```env
LOG_LEVEL=DEBUG
LOG_FILE=logs/app.log
LOG_JSON_FORMAT=false  # 便于阅读
```

**生产环境：**
```env
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
LOG_JSON_FORMAT=true  # 便于日志分析
```

### 3. 安全配置

**保护敏感信息：**
1. ✓ 永远不要提交 `.env` 文件到Git
2. ✓ 使用强密码
3. ✓ 限制数据库用户权限
4. ✓ 生产环境使用环境变量而非文件

**权限建议：**
```sql
-- 创建专用数据库用户
CREATE USER 'stock_app'@'localhost' IDENTIFIED BY 'strong_password';

-- 授予必要权限
GRANT SELECT, INSERT, UPDATE ON stock_db.* TO 'stock_app'@'localhost';

-- 刷新权限
FLUSH PRIVILEGES;
```

## Git配置

### .gitignore说明

系统已配置 `.gitignore` 文件，忽略以下内容：

**敏感信息：**
- `.env` - 环境变量配置
- `*.log` - 日志文件

**运行时文件：**
- `logs/` - 日志目录
- `checkpoints/` - 模型检查点
- `__pycache__/` - Python缓存

**开发环境：**
- `venv/` - 虚拟环境
- `.vscode/` - IDE配置
- `.idea/` - IDE配置

**数据文件：**
- `data/` - 数据目录
- `*.csv` - CSV文件
- `*.pth` - PyTorch模型文件

**完整列表见 `.gitignore` 文件**

## 配置验证

### 验证步骤

1. **检查配置文件**
   ```bash
   python check_config.py
   ```

2. **测试数据库连接**
   ```bash
   python -c "from src.database.connection import engine; engine.connect(); print('连接成功')"
   ```

3. **初始化AI表**
   ```bash
   python src/database/init_db.py
   ```

4. **运行快速测试**
   ```bash
   python quick_test.py
   ```

### 常见问题

**Q: 数据库连接失败**

A: 检查以下项：
1. MySQL服务是否启动
2. .env文件中的配置是否正确
3. 数据库是否已创建
4. 用户权限是否正确

**Q: 配置导入失败**

A: 检查：
1. 是否安装了 `pydantic-settings`
2. .env文件格式是否正确
3. 配置项名称是否正确

**Q: Redis连接失败**

A: 
1. 训练模块不需要Redis，可以忽略
2. API服务需要Redis，检查Redis是否启动
3. 检查Redis连接配置

## 配置模板

### 最小配置（仅训练）

```env
# 数据库配置
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=stock_db

# 应用配置
APP_ENV=development
LOG_LEVEL=INFO
```

### 完整配置（训练+API）

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

## 相关文档

- [快速开始指南](QUICK_START.md)
- [测试指南](TESTING_GUIDE.md)
- [API文档](API_DOCUMENTATION.md)
