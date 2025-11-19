# Design Document - A股市场AI模型系统

## Overview

本系统是一个端到端的A股市场智能分析平台，采用微服务架构设计，集成数据采集、特征工程、模型训练、预测服务和可视化展示等功能模块。系统使用深度学习技术（LSTM、GRU、Transformer）进行时间序列预测，并结合传统技术分析指标提供综合投资决策支持。

### Technology Stack

- **Backend**: Python 3.10+, FastAPI
- **Machine Learning**: PyTorch, scikit-learn, pandas, numpy
- **Data Storage**: MySQL (结构化数据 - 已有), Redis (缓存), MinIO (模型文件存储)
- **Data Collection**: 使用现有数据库数据 (stock_basic_info, stock_kline_data)
- **Frontend**: React, TypeScript, ECharts/TradingView
- **Task Queue**: Celery + Redis
- **Monitoring**: Prometheus + Grafana

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Stock Search │  │ Chart Display│  │ Model Config │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────────┬────────────────────────────────┘
                             │ REST API
┌────────────────────────────┼────────────────────────────────┐
│                    API Gateway (FastAPI)                     │
└────────────────────────────┬────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
│ Data Service   │  │ Prediction     │  │ Training       │
│                │  │ Service        │  │ Service        │
│ - Data Fetch   │  │ - Inference    │  │ - Model Train  │
│ - Validation   │  │ - Indicators   │  │ - Evaluation   │
│ - Storage      │  │ - Cache        │  │ - Versioning   │
└───────┬────────┘  └───────┬────────┘  └───────┬────────┘
        │                   │                    │
        └───────────────────┼────────────────────┘
                            │
        ┌───────────────────┼────────────────────┐
        │                   │                    │
┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
│  PostgreSQL    │  │     Redis      │  │     MinIO      │
│  (Stock Data)  │  │    (Cache)     │  │  (ML Models)   │
└────────────────┘  └────────────────┘  └────────────────┘
```

### Data Flow

1. **数据采集流程**: Celery定时任务 → akshare API → 数据验证 → PostgreSQL存储
2. **训练流程**: 训练请求 → 数据加载 → 特征工程 → 模型训练 → 评估 → MinIO保存
3. **预测流程**: 预测请求 → Redis缓存检查 → 模型推理 → 技术指标计算 → 返回结果

## Components and Interfaces

### 1. Data Collection Service

**职责**: 从数据源获取A股市场数据并进行预处理

**核心类**:

```python
class StockDataLoader:
    """股票数据加载器（从现有MySQL数据库）"""
    
    def load_kline_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从stock_kline_data表加载K线数据"""
        pass
    
    def load_stock_info(self, symbol: str) -> Dict:
        """从stock_basic_info表加载股票基本信息"""
        pass
    
    def get_all_active_stocks(self) -> List[str]:
        """获取所有活跃股票代码列表（is_active=1）"""
        pass
    
    def validate_data_completeness(self, symbol: str, start_date: str, end_date: str) -> bool:
        """验证指定时间段数据完整性"""
        pass

class DataValidator:
    """数据验证器"""
    
    def check_missing_values(self, df: pd.DataFrame) -> bool:
        pass
    
    def check_price_anomalies(self, df: pd.DataFrame) -> List[str]:
        """检测价格异常（如涨跌幅超过限制）"""
        pass
```

**API接口**:

```
GET /api/v1/data/stocks
- Query: market, industry, is_active
- Response: { "stocks": [...], "count": 5000 }

GET /api/v1/data/stocks/{symbol}/kline
- Query: start_date, end_date
- Response: { "symbol": "000001", "data": [...], "count": 1000 }

GET /api/v1/data/stocks/{symbol}/info
- Response: { "symbol": "000001", "name": "平安银行", "industry": "银行", ... }
```

### 2. Feature Engineering Module

**职责**: 从原始数据生成模型训练所需的特征

**核心功能**:

```python
class FeatureEngineer:
    """特征工程"""
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成技术指标特征"""
        # MA, EMA, MACD, RSI, Bollinger Bands, KDJ等
        pass
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成价格相关特征"""
        # 涨跌幅、振幅、价格变化率等
        pass
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成成交量特征"""
        # 量比、换手率等
        pass
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征标准化"""
        pass
    
    def create_sequences(self, df: pd.DataFrame, seq_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """创建时间序列样本（滑动窗口）"""
        pass
```

### 3. Training Service

**职责**: 训练和评估机器学习模型

**模型架构**:

```python
class LSTMModel(nn.Module):
    """LSTM预测模型"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

class TransformerModel(nn.Module):
    """Transformer预测模型"""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, output_size: int):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        transformer_out = self.transformer(x)
        predictions = self.fc(transformer_out[:, -1, :])
        return predictions
```

**训练管理**:

```python
class ModelTrainer:
    """模型训练器"""
    
    def train(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int, learning_rate: float) -> Dict:
        """训练模型"""
        pass
    
    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> Dict:
        """评估模型性能"""
        # 返回 MAE, RMSE, MAPE, 方向准确率等指标
        pass
    
    def save_checkpoint(self, model: nn.Module, epoch: int, metrics: Dict) -> str:
        """保存模型检查点"""
        pass
    
    def load_checkpoint(self, checkpoint_path: str) -> nn.Module:
        """加载模型检查点"""
        pass
```

**API接口**:

```
POST /api/v1/training/start
- Body: { "stock_code": "000001", "model_type": "lstm", "config": {...} }
- Response: { "task_id": "uuid", "status": "started" }

GET /api/v1/training/status/{task_id}
- Response: { "status": "training", "progress": 45, "current_epoch": 45, "metrics": {...} }
```

### 4. Prediction Service

**职责**: 使用训练好的模型生成预测结果

**核心类**:

```python
class PredictionEngine:
    """预测引擎"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = self.load_model(model_path)
        self.device = device
    
    def predict(self, stock_code: str, days: int = 5) -> Dict:
        """生成未来N天的价格预测"""
        # 返回预测价格、置信区间、趋势方向
        pass
    
    def predict_with_confidence(self, stock_code: str, days: int = 5, 
                               confidence_level: float = 0.95) -> Dict:
        """生成带置信区间的预测"""
        pass
    
    def batch_predict(self, stock_codes: List[str], days: int = 5) -> Dict:
        """批量预测多只股票"""
        pass

class TechnicalIndicatorCalculator:
    """技术指标计算器"""
    
    def calculate_ma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """计算移动平均线"""
        pass
    
    def calculate_macd(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算MACD指标"""
        pass
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """计算RSI指标"""
        pass
    
    def calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20) -> Tuple:
        """计算布林带"""
        pass
```

**API接口**:

```
POST /api/v1/prediction/predict
- Body: { "stock_code": "000001", "days": 5, "model_version": "v1.2" }
- Response: {
    "stock_code": "000001",
    "predictions": [
      {"date": "2025-11-13", "price": 15.23, "confidence_lower": 14.89, "confidence_upper": 15.57},
      ...
    ],
    "trend": "upward",
    "confidence_score": 0.85
  }

GET /api/v1/prediction/indicators/{stock_code}
- Query: indicators=ma,macd,rsi
- Response: { "ma_5": [...], "ma_10": [...], "macd": {...}, "rsi": [...] }
```

### 5. Frontend Application

**主要页面**:

1. **股票搜索页**: 搜索和选择股票
2. **数据展示页**: K线图、技术指标、历史数据
3. **预测结果页**: AI预测曲线、置信区间、趋势分析
4. **模型管理页**: 模型训练、性能对比、版本管理
5. **系统监控页**: 数据更新状态、模型运行状态

**核心组件**:

```typescript
interface StockChartProps {
  stockCode: string;
  historicalData: PriceData[];
  predictions: PredictionData[];
  indicators: TechnicalIndicators;
}

interface PredictionData {
  date: string;
  price: number;
  confidenceLower: number;
  confidenceUpper: number;
}

interface TechnicalIndicators {
  ma5: number[];
  ma10: number[];
  macd: MACDData;
  rsi: number[];
  bollingerBands: BollingerData;
}
```

## Data Models

### Database Schema

**现有表（已有数据）**:

```sql
-- 股票基本信息表 (现有)
CREATE TABLE `stock_basic_info` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `symbol` varchar(10) NOT NULL COMMENT '股票代码',
  `name` varchar(50) NOT NULL COMMENT '股票名称',
  `exchange` varchar(10) DEFAULT NULL COMMENT '交易所',
  `market` varchar(10) NOT NULL COMMENT '交易市场',
  `stock_type` varchar(10) NOT NULL COMMENT '股票类型',
  `industry` varchar(30) DEFAULT NULL COMMENT '所属行业(证监会)',
  `list_date` date DEFAULT NULL COMMENT '上市日期',
  `delist_date` date DEFAULT NULL COMMENT '退市日期',
  `is_active` int NOT NULL COMMENT '是否上市交易',
  `total_share` decimal(20,2) DEFAULT NULL COMMENT '总股本(股)',
  `float_share` decimal(20,2) DEFAULT NULL COMMENT '流通股本(股)',
  `total_market_value` decimal(20,2) DEFAULT NULL COMMENT '总市值(元)',
  `float_market_value` decimal(20,2) DEFAULT NULL COMMENT '流通市值(元)',
  `created_at` datetime NOT NULL COMMENT '创建时间',
  `updated_at` datetime NOT NULL COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `symbol` (`symbol`),
  KEY `stock_basic_info_market_idx` (`market`),
  KEY `stock_basic_info_industry_idx` (`industry`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- 日线数据表 (现有)
CREATE TABLE `stock_kline_data` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `symbol` varchar(20) NOT NULL COMMENT '股票代码',
  `trade_date` date NOT NULL COMMENT '交易日期',
  `open` decimal(10,3) NOT NULL COMMENT '开盘价',
  `high` decimal(10,3) NOT NULL COMMENT '最高价',
  `low` decimal(10,3) NOT NULL COMMENT '最低价',
  `close` decimal(10,3) NOT NULL COMMENT '收盘价',
  `pre_close` decimal(10,3) DEFAULT NULL COMMENT '昨收价',
  `change` decimal(10,3) DEFAULT NULL COMMENT '涨跌额',
  `pct_chg` decimal(8,3) DEFAULT NULL COMMENT '涨跌幅(%)',
  `vol` bigint DEFAULT NULL COMMENT '成交量(手)',
  `amount` decimal(15,2) DEFAULT NULL COMMENT '成交额(千元)',
  `turnover_rate` decimal(8,3) DEFAULT NULL COMMENT '换手率(%)',
  `pe` decimal(10,3) DEFAULT NULL COMMENT '市盈率',
  `pb` decimal(10,3) DEFAULT NULL COMMENT '市净率',
  `total_mv` decimal(15,2) DEFAULT NULL COMMENT '总市值(万元)',
  `circ_mv` decimal(15,2) DEFAULT NULL COMMENT '流通市值(万元)',
  `limit_status` int DEFAULT NULL COMMENT '涨跌停状态 0正常 1涨停 -1跌停',
  `created_at` datetime DEFAULT NULL,
  `updated_at` datetime DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_symbol_date` (`symbol`,`trade_date`),
  KEY `idx_trade_date` (`trade_date`),
  KEY `idx_symbol` (`symbol`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
```

**新增表（AI模型相关）**:

```sql
-- 模型信息表
CREATE TABLE `ai_models` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `model_id` varchar(36) NOT NULL COMMENT '模型UUID',
  `model_name` varchar(100) NOT NULL COMMENT '模型名称',
  `model_type` varchar(50) NOT NULL COMMENT '模型类型: lstm, gru, transformer',
  `version` varchar(20) NOT NULL COMMENT '版本号',
  `symbol` varchar(10) DEFAULT NULL COMMENT '股票代码(NULL表示通用模型)',
  `training_start_date` date DEFAULT NULL COMMENT '训练数据起始日期',
  `training_end_date` date DEFAULT NULL COMMENT '训练数据结束日期',
  `hyperparameters` json DEFAULT NULL COMMENT '超参数配置',
  `performance_metrics` json DEFAULT NULL COMMENT '性能指标',
  `model_path` varchar(255) DEFAULT NULL COMMENT '模型文件路径',
  `status` varchar(20) NOT NULL COMMENT '状态: training, completed, failed',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_model_id` (`model_id`),
  KEY `idx_symbol` (`symbol`),
  KEY `idx_status` (`status`),
  KEY `idx_model_type` (`model_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- 预测结果表
CREATE TABLE `ai_predictions` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `model_id` varchar(36) NOT NULL COMMENT '模型ID',
  `symbol` varchar(10) NOT NULL COMMENT '股票代码',
  `prediction_date` date NOT NULL COMMENT '预测生成日期',
  `target_date` date NOT NULL COMMENT '预测目标日期',
  `predicted_close` decimal(10,3) NOT NULL COMMENT '预测收盘价',
  `confidence_lower` decimal(10,3) DEFAULT NULL COMMENT '置信区间下限',
  `confidence_upper` decimal(10,3) DEFAULT NULL COMMENT '置信区间上限',
  `actual_close` decimal(10,3) DEFAULT NULL COMMENT '实际收盘价',
  `prediction_error` decimal(10,3) DEFAULT NULL COMMENT '预测误差',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_model_symbol_dates` (`model_id`, `symbol`, `prediction_date`, `target_date`),
  KEY `idx_symbol_target` (`symbol`, `target_date`),
  KEY `idx_prediction_date` (`prediction_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- 技术指标缓存表 (可选，用于持久化计算结果)
CREATE TABLE `technical_indicators` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `symbol` varchar(10) NOT NULL COMMENT '股票代码',
  `trade_date` date NOT NULL COMMENT '交易日期',
  `ma5` decimal(10,3) DEFAULT NULL COMMENT '5日均线',
  `ma10` decimal(10,3) DEFAULT NULL COMMENT '10日均线',
  `ma20` decimal(10,3) DEFAULT NULL COMMENT '20日均线',
  `ma60` decimal(10,3) DEFAULT NULL COMMENT '60日均线',
  `ema12` decimal(10,3) DEFAULT NULL COMMENT '12日指数移动平均',
  `ema26` decimal(10,3) DEFAULT NULL COMMENT '26日指数移动平均',
  `macd_dif` decimal(10,3) DEFAULT NULL COMMENT 'MACD DIF',
  `macd_dea` decimal(10,3) DEFAULT NULL COMMENT 'MACD DEA',
  `macd_bar` decimal(10,3) DEFAULT NULL COMMENT 'MACD柱',
  `rsi6` decimal(10,3) DEFAULT NULL COMMENT '6日RSI',
  `rsi12` decimal(10,3) DEFAULT NULL COMMENT '12日RSI',
  `rsi24` decimal(10,3) DEFAULT NULL COMMENT '24日RSI',
  `boll_upper` decimal(10,3) DEFAULT NULL COMMENT '布林带上轨',
  `boll_mid` decimal(10,3) DEFAULT NULL COMMENT '布林带中轨',
  `boll_lower` decimal(10,3) DEFAULT NULL COMMENT '布林带下轨',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_symbol_date` (`symbol`, `trade_date`),
  KEY `idx_trade_date` (`trade_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
```

### Redis Cache Structure

```
# 预测结果缓存 (TTL: 1小时)
prediction:{stock_code}:{model_version} -> JSON

# 技术指标缓存 (TTL: 30分钟)
indicators:{stock_code}:{date} -> JSON

# 股票列表缓存 (TTL: 24小时)
stocks:all -> List[stock_code]

# 模型元数据缓存 (TTL: 1小时)
model:{model_id}:metadata -> JSON
```

## Error Handling

### Error Categories

1. **数据采集错误**
   - 网络连接失败: 重试机制（最多3次，指数退避）
   - API限流: 等待并重试
   - 数据格式错误: 记录日志，跳过该条数据

2. **模型训练错误**
   - 内存不足: 减小batch size，使用梯度累积
   - 训练发散: 降低学习率，使用梯度裁剪
   - 数据不足: 返回错误信息，要求更多历史数据

3. **预测服务错误**
   - 模型加载失败: 尝试加载备用模型
   - 输入数据异常: 返回400错误和详细信息
   - 超时: 设置5秒超时，返回503错误

### Error Response Format

```json
{
  "error": {
    "code": "MODEL_TRAINING_FAILED",
    "message": "模型训练失败：数据不足",
    "details": {
      "required_samples": 1000,
      "actual_samples": 500
    },
    "timestamp": "2025-11-12T10:30:00Z"
  }
}
```

### Logging Strategy

- **级别**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **格式**: JSON结构化日志
- **存储**: 本地文件 + ELK Stack（生产环境）
- **关键事件**: 数据采集、模型训练开始/结束、预测请求、错误异常

## Testing Strategy

### 1. Unit Testing

**测试范围**:
- 数据验证逻辑
- 特征工程函数
- 技术指标计算
- 模型前向传播

**工具**: pytest, pytest-cov

**示例**:
```python
def test_feature_engineer_ma_calculation():
    """测试移动平均线计算"""
    prices = np.array([10, 11, 12, 13, 14])
    ma = FeatureEngineer().calculate_ma(prices, period=3)
    assert ma[-1] == 13.0  # (12+13+14)/3

def test_data_validator_missing_values():
    """测试缺失值检测"""
    df = pd.DataFrame({'close': [10, np.nan, 12]})
    validator = DataValidator()
    assert validator.check_missing_values(df) == False
```

### 2. Integration Testing

**测试范围**:
- API端点测试
- 数据库操作测试
- 服务间通信测试

**工具**: pytest, httpx, testcontainers

**示例**:
```python
@pytest.mark.asyncio
async def test_prediction_api():
    """测试预测API"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/v1/prediction/predict", 
                                     json={"stock_code": "000001", "days": 5})
        assert response.status_code == 200
        assert "predictions" in response.json()
```

### 3. Model Testing

**测试范围**:
- 模型性能基准测试
- 过拟合检测
- 预测准确性验证

**评估指标**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- 方向准确率 (Direction Accuracy)

**测试数据**: 使用最近1年数据作为测试集，不参与训练

### 4. Performance Testing

**测试场景**:
- 并发预测请求 (100 QPS)
- 大批量数据采集
- 模型推理延迟

**工具**: Locust, Apache JMeter

**性能目标**:
- 预测API响应时间 < 5秒 (P95)
- 数据采集吞吐量 > 1000条/秒
- 系统可用性 > 99.5%

### 5. End-to-End Testing

**测试流程**:
1. 采集指定股票的历史数据
2. 触发模型训练
3. 等待训练完成
4. 请求预测结果
5. 验证预测数据格式和合理性

**工具**: Selenium (前端), pytest (后端)

## Security Considerations

1. **API安全**: JWT认证，Rate Limiting
2. **数据加密**: 敏感数据AES-256加密存储
3. **输入验证**: 严格验证所有用户输入，防止SQL注入
4. **访问控制**: RBAC权限管理
5. **审计日志**: 记录所有关键操作

## Deployment Architecture

```
┌─────────────────────────────────────────┐
│          Load Balancer (Nginx)          │
└────────────┬────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼────┐      ┌────▼───┐
│ API    │      │ API    │
│ Server │      │ Server │
│   1    │      │   2    │
└───┬────┘      └────┬───┘
    │                │
    └────────┬───────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼────┐      ┌────▼───┐
│ Celery │      │ Celery │
│ Worker │      │ Worker │
│   1    │      │   2    │
└────────┘      └────────┘
```

**容器化**: Docker + Docker Compose (开发), Kubernetes (生产)

**扩展性**: 
- API服务: 水平扩展
- Worker服务: 根据任务队列长度自动扩展
- 数据库: 主从复制，读写分离
