# 数据源控制指南

## 如何区分缓存和MySQL？

系统提供了灵活的数据源控制机制，你可以明确指定使用缓存还是MySQL。

## 方法1：使用不同的加载器类

### 直接使用MySQL（不使用缓存）

```python
from src.data.loader import StockDataLoader

# 使用原始加载器，直接从MySQL读取
loader = StockDataLoader()
df = loader.load_kline_data("000001", "2021-01-01", "2024-12-31")

# 每次都从MySQL加载，不使用缓存
```

**特点：**
- ✓ 总是获取最新数据
- ✓ 不依赖缓存
- ✗ 速度较慢
- ✗ 增加数据库负载

### 使用缓存（优先缓存，缓存不存在时从MySQL加载）

```python
from src.data.cached_loader import ParquetDataLoader

# 使用缓存加载器
loader = ParquetDataLoader(cache_dir="data/parquet")
df = loader.load_kline_data("000001", "2021-01-01", "2024-12-31", use_cache=True)

# 自动判断：
# 1. 如果缓存存在且未过期 → 从缓存加载
# 2. 如果缓存不存在或过期 → 从MySQL加载并缓存
```

**特点：**
- ✓ 自动管理缓存
- ✓ 首次慢，后续快
- ✓ 智能降级

## 方法2：使用参数控制

### use_cache 参数

```python
from src.data.cached_loader import ParquetDataLoader

loader = ParquetDataLoader()

# 使用缓存（默认）
df = loader.load_kline_data("000001", "2021-01-01", "2024-12-31", use_cache=True)

# 不使用缓存，强制从MySQL加载
df = loader.load_kline_data("000001", "2021-01-01", "2024-12-31", use_cache=False)
```

### force_refresh 参数

```python
from src.data.cached_loader import ParquetDataLoader

loader = ParquetDataLoader()

# 强制刷新：忽略现有缓存，从MySQL重新加载并更新缓存
df = loader.load_kline_data(
    "000001",
    "2021-01-01",
    "2024-12-31",
    use_cache=True,
    force_refresh=True  # 强制刷新
)
```

## 方法3：通过配置文件控制

### 创建配置文件

```python
# config/training_config.py

class TrainingConfig:
    # 数据源配置
    USE_CACHE = True  # 是否使用缓存
    CACHE_DIR = "data/parquet"  # 缓存目录
    CACHE_TTL = 86400  # 缓存有效期（秒）
    
    # 环境配置
    ENV = "development"  # development / production
    
    @classmethod
    def get_loader(cls):
        """根据配置获取加载器"""
        if cls.USE_CACHE:
            from src.data.cached_loader import ParquetDataLoader
            return ParquetDataLoader(
                cache_dir=cls.CACHE_DIR,
                cache_ttl=cls.CACHE_TTL
            )
        else:
            from src.data.loader import StockDataLoader
            return StockDataLoader()
```

### 使用配置

```python
from config.training_config import TrainingConfig

# 根据配置自动选择数据源
loader = TrainingConfig.get_loader()
df = loader.load_kline_data("000001", "2021-01-01", "2024-12-31")
```

## 实际使用场景

### 场景1：开发和调试（使用缓存）

```python
from src.data.cached_loader import ParquetDataLoader

# 开发时使用缓存，加快迭代速度
loader = ParquetDataLoader()
df = loader.load_kline_data("000001", "2021-01-01", "2024-12-31", use_cache=True)

print("使用缓存，速度快，适合反复调试")
```

### 场景2：生产训练（使用缓存）

```python
from src.data.cached_loader import ParquetDataLoader

# 生产环境使用缓存，提升性能
loader = ParquetDataLoader(cache_dir="/data/cache/parquet")
df = loader.load_kline_data("000001", "2021-01-01", "2024-12-31", use_cache=True)

print("使用缓存，降低数据库负载")
```

### 场景3：实时预测（直接MySQL）

```python
from src.data.loader import StockDataLoader

# 实时预测需要最新数据，不使用缓存
loader = StockDataLoader()
df = loader.load_kline_data("000001", "2021-01-01", "2024-12-31")

print("直接从MySQL读取，确保数据最新")
```

### 场景4：数据验证（强制刷新）

```python
from src.data.cached_loader import ParquetDataLoader

# 验证数据时，强制从MySQL重新加载
loader = ParquetDataLoader()
df = loader.load_kline_data(
    "000001",
    "2021-01-01",
    "2024-12-31",
    use_cache=True,
    force_refresh=True  # 强制刷新缓存
)

print("强制从MySQL加载，更新缓存")
```

## 自动判断逻辑

### ParquetDataLoader 的工作流程

```python
def load_kline_data(self, symbol, start_date, end_date, use_cache=True, force_refresh=False):
    """
    智能加载数据
    
    决策流程：
    1. 如果 use_cache=False → 直接从MySQL加载
    2. 如果 force_refresh=True → 从MySQL加载并更新缓存
    3. 如果缓存文件存在：
       a. 检查缓存是否过期
       b. 未过期 → 从缓存加载
       c. 已过期 → 从MySQL加载并更新缓存
    4. 如果缓存文件不存在 → 从MySQL加载并创建缓存
    """
    
    cache_file = self.cache_dir / f"{symbol}.parquet"
    
    # 情况1：不使用缓存
    if not use_cache:
        return self._load_from_mysql(symbol, start_date, end_date)
    
    # 情况2：强制刷新
    if force_refresh:
        df = self._load_from_mysql(symbol, start_date, end_date)
        self._save_to_cache(df, cache_file)
        return df
    
    # 情况3：检查缓存
    if cache_file.exists():
        cache_age = time.time() - cache_file.stat().st_mtime
        
        if cache_age < self.cache_ttl:
            # 缓存有效，从缓存加载
            return self._load_from_cache(cache_file, start_date, end_date)
        else:
            # 缓存过期，从MySQL加载
            df = self._load_from_mysql(symbol, start_date, end_date)
            self._save_to_cache(df, cache_file)
            return df
    
    # 情况4：缓存不存在
    df = self._load_from_mysql(symbol, start_date, end_date)
    self._save_to_cache(df, cache_file)
    return df
```

## 日志输出区分

系统会自动记录数据来源：

```python
# 从缓存加载
logger.info(f"从缓存加载 {symbol}: {len(df)} 条记录, 耗时 0.05秒")

# 从MySQL加载
logger.info(f"从MySQL加载 {symbol}: {len(df)} 条记录, 耗时 0.5秒")
```

**运行时输出示例：**
```
INFO - 从缓存加载 000001: 750 条记录, 耗时 0.045秒
INFO - 从MySQL加载 600519: 750 条记录, 耗时 0.523秒
INFO - 缓存已保存: data/parquet/600519.parquet
```

## 推荐使用模式

### 模式1：智能模式（推荐）

```python
from src.data.cached_loader import ParquetDataLoader

# 使用缓存加载器，自动管理缓存
loader = ParquetDataLoader()

# 默认使用缓存，自动判断
df = loader.load_kline_data("000001", "2021-01-01", "2024-12-31")
```

**优点：**
- 自动管理缓存
- 首次慢，后续快
- 缓存过期自动刷新
- 适合大多数场景

### 模式2：纯MySQL模式

```python
from src.data.loader import StockDataLoader

# 直接使用MySQL加载器
loader = StockDataLoader()

# 总是从MySQL加载
df = loader.load_kline_data("000001", "2021-01-01", "2024-12-31")
```

**优点：**
- 总是最新数据
- 不依赖缓存
- 适合实时场景

### 模式3：混合模式

```python
from src.data.loader import StockDataLoader
from src.data.cached_loader import ParquetDataLoader

# 根据场景选择
def get_loader(use_cache=True):
    if use_cache:
        return ParquetDataLoader()
    else:
        return StockDataLoader()

# 训练时使用缓存
train_loader = get_loader(use_cache=True)
df_train = train_loader.load_kline_data("000001", "2021-01-01", "2024-12-31")

# 实时预测使用MySQL
predict_loader = get_loader(use_cache=False)
df_predict = predict_loader.load_kline_data("000001", "2024-12-01", "2024-12-31")
```

## 环境变量控制

### 通过环境变量配置

```bash
# .env 文件
USE_DATA_CACHE=true
DATA_CACHE_DIR=data/parquet
DATA_CACHE_TTL=86400
```

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

USE_DATA_CACHE = os.getenv("USE_DATA_CACHE", "true").lower() == "true"
DATA_CACHE_DIR = os.getenv("DATA_CACHE_DIR", "data/parquet")
DATA_CACHE_TTL = int(os.getenv("DATA_CACHE_TTL", "86400"))
```

```python
# 使用配置
from src.config import USE_DATA_CACHE, DATA_CACHE_DIR
from src.data.loader import StockDataLoader
from src.data.cached_loader import ParquetDataLoader

if USE_DATA_CACHE:
    loader = ParquetDataLoader(cache_dir=DATA_CACHE_DIR)
else:
    loader = StockDataLoader()

df = loader.load_kline_data("000001", "2021-01-01", "2024-12-31")
```

## 命令行参数控制

```python
# train.py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use-cache", action="store_true", help="使用缓存")
parser.add_argument("--no-cache", action="store_true", help="不使用缓存")
args = parser.parse_args()

# 根据参数选择加载器
if args.no_cache:
    from src.data.loader import StockDataLoader
    loader = StockDataLoader()
else:
    from src.data.cached_loader import ParquetDataLoader
    loader = ParquetDataLoader()

df = loader.load_kline_data("000001", "2021-01-01", "2024-12-31")
```

**使用方法：**
```bash
# 使用缓存
python train.py --use-cache

# 不使用缓存
python train.py --no-cache
```

## 检查数据来源

### 方法1：查看日志

```python
import logging
logging.basicConfig(level=logging.INFO)

from src.data.cached_loader import ParquetDataLoader

loader = ParquetDataLoader()
df = loader.load_kline_data("000001", "2021-01-01", "2024-12-31")

# 日志会显示：
# INFO - 从缓存加载 000001: 750 条记录, 耗时 0.045秒
# 或
# INFO - 从MySQL加载 000001: 750 条记录, 耗时 0.523秒
```

### 方法2：检查缓存文件

```python
from pathlib import Path

cache_file = Path("data/parquet/000001.parquet")

if cache_file.exists():
    print(f"缓存存在: {cache_file}")
    print(f"文件大小: {cache_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"修改时间: {cache_file.stat().st_mtime}")
else:
    print("缓存不存在，将从MySQL加载")
```

### 方法3：性能对比

```python
import time

# 测试加载时间
start_time = time.time()
df = loader.load_kline_data("000001", "2021-01-01", "2024-12-31")
elapsed_time = time.time() - start_time

if elapsed_time < 0.1:
    print(f"从缓存加载（耗时 {elapsed_time:.3f}秒）")
else:
    print(f"从MySQL加载（耗时 {elapsed_time:.3f}秒）")
```

## 总结

### 快速参考

| 需求 | 使用方法 | 代码 |
|------|---------|------|
| 总是使用MySQL | StockDataLoader | `loader = StockDataLoader()` |
| 智能使用缓存 | ParquetDataLoader | `loader = ParquetDataLoader()` |
| 强制使用MySQL | use_cache=False | `load_kline_data(..., use_cache=False)` |
| 强制刷新缓存 | force_refresh=True | `load_kline_data(..., force_refresh=True)` |

### 推荐配置

**开发环境：**
```python
# 使用缓存，加快开发速度
loader = ParquetDataLoader()
df = loader.load_kline_data(..., use_cache=True)
```

**生产环境：**
```python
# 使用缓存，降低数据库负载
loader = ParquetDataLoader(cache_dir="/data/cache")
df = loader.load_kline_data(..., use_cache=True)
```

**实时预测：**
```python
# 直接MySQL，确保数据最新
loader = StockDataLoader()
df = loader.load_kline_data(...)
```

**数据验证：**
```python
# 强制刷新，验证数据
loader = ParquetDataLoader()
df = loader.load_kline_data(..., force_refresh=True)
```

系统提供了灵活的控制机制，你可以根据具体场景选择最合适的方式！
