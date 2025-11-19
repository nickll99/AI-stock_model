# 数据预热详解

## 什么是数据预热？

数据预热（Data Preheating）是指在正式训练之前，**提前将数据从MySQL数据库加载到本地磁盘缓存**的过程。

### 类比理解

就像冬天开车前需要预热发动机一样：

**传统方式（每次从MySQL加载）：**
```
训练1: MySQL → 内存 → 训练 (慢)
训练2: MySQL → 内存 → 训练 (慢)
训练3: MySQL → 内存 → 训练 (慢)
```

**数据预热方式：**
```
预热阶段: MySQL → 本地缓存 (一次性，慢)
训练1: 本地缓存 → 内存 → 训练 (快)
训练2: 本地缓存 → 内存 → 训练 (快)
训练3: 本地缓存 → 内存 → 训练 (快)
```

## 为什么需要数据预热？

### 问题场景

**场景1：超参数调优**
```python
# 需要训练100次，每次都从MySQL加载相同数据
for learning_rate in [0.001, 0.005, 0.01, ...]:
    df = load_from_mysql("000001")  # 重复加载100次！
    train_model(df, learning_rate)
```

**场景2：批量训练**
```python
# 训练5000只股票，每只都要查询MySQL
for symbol in all_5000_stocks:
    df = load_from_mysql(symbol)  # 5000次数据库查询！
    train_model(df)
```

**场景3：模型对比**
```python
# 对比LSTM、GRU、Transformer三种模型
for model_type in ['lstm', 'gru', 'transformer']:
    df = load_from_mysql("000001")  # 重复加载3次！
    train_model(df, model_type)
```

### 性能瓶颈

1. **网络IO延迟** - 每次查询MySQL都有网络往返
2. **数据库负载** - 大量查询增加数据库压力
3. **重复计算** - 技术指标每次都要重新计算
4. **时间浪费** - 大部分时间花在数据加载上

## 数据预热的具体过程

### 第一步：K线数据预热

**目标：** 将MySQL中的原始K线数据缓存到本地Parquet文件

```
MySQL数据库                    本地磁盘缓存
┌─────────────────┐           ┌─────────────────┐
│ stock_kline_data│           │ data/parquet/   │
│                 │           │                 │
│ 000001: 750条   │  ────>    │ 000001.parquet  │
│ 600519: 750条   │  ────>    │ 600519.parquet  │
│ 000002: 750条   │  ────>    │ 000002.parquet  │
│ ...             │  ────>    │ ...             │
└─────────────────┘           └─────────────────┘
```

**执行过程：**
```python
# 1. 从MySQL加载数据
df = mysql_loader.load_kline_data("000001", "2021-01-01", "2024-12-31")

# 2. 保存为Parquet格式（列式存储，压缩）
df.to_parquet("data/parquet/000001.parquet", compression='snappy')
```

**文件格式：**
```
data/parquet/
├── 000001.parquet  (2MB, 包含750条记录)
├── 600519.parquet  (2MB)
├── 000002.parquet  (2MB)
└── ...
```

### 第二步：特征数据预热

**目标：** 预计算技术指标和特征，缓存到本地

```
K线数据                特征工程              特征缓存
┌──────────┐         ┌──────────┐         ┌──────────────┐
│ 000001   │         │ 计算MA   │         │ data/features│
│ OHLCV    │  ────>  │ 计算MACD │  ────>  │ 000001_      │
│ 750条    │         │ 计算RSI  │         │ features.    │
│          │         │ ...      │         │ parquet      │
└──────────┘         └──────────┘         └──────────────┘
```

**执行过程：**
```python
# 1. 从缓存加载K线数据
df = parquet_loader.load_kline_data("000001", "2021-01-01", "2024-12-31")

# 2. 计算技术指标（耗时操作）
df_features = feature_engineer.create_technical_indicators(df)
df_features = feature_engineer.create_price_features(df_features)
df_features = feature_engineer.create_volume_features(df_features)

# 3. 保存特征缓存
df_features.to_parquet("data/features/000001_features.parquet")
```

**特征包含：**
- 原始数据：open, high, low, close, volume
- 技术指标：MA5, MA10, MA20, MACD, RSI, 布林带, KDJ
- 价格特征：涨跌幅、振幅、价格变化率
- 成交量特征：量比、换手率

**文件格式：**
```
data/features/
├── 000001_features.parquet  (3MB, 包含45个特征)
├── 600519_features.parquet  (3MB)
├── 000002_features.parquet  (3MB)
└── ...
```

## 使用数据预热脚本

### 基本用法

```bash
# 预热所有活跃股票的数据（推荐）
python scripts/prepare_training_data.py --symbols all --start-date 2021-01-01 --end-date 2024-12-31
```

**执行过程：**
```
======================================================================
  预热K线数据
======================================================================

股票数量: 5000
日期范围: 2021-01-01 至 2024-12-31
缓存目录: data/parquet

[1/5000] 000001: 750 条记录已缓存
[2/5000] 600519: 750 条记录已缓存
[3/5000] 000002: 750 条记录已缓存
...
[5000/5000] 688599: 750 条记录已缓存

======================================================================
K线数据预热完成
  成功: 4800
  失败: 200
  总记录数: 3,600,000
  缓存大小: 9,600.00 MB
======================================================================

======================================================================
  预热特征数据
======================================================================

股票数量: 5000
特征缓存目录: data/features

[1/5000] 000001: 750 条记录, 45 个特征已缓存
[2/5000] 600519: 750 条记录, 45 个特征已缓存
...

======================================================================
特征数据预热完成
  成功: 4800
  失败: 200
  缓存大小: 14,400.00 MB
======================================================================
```

### 高级用法

#### 1. 仅预热指定股票

```bash
# 预热单只股票
python scripts/prepare_training_data.py --symbols 000001 --start-date 2021-01-01

# 预热多只股票
python scripts/prepare_training_data.py --symbols 000001,600519,000002 --start-date 2021-01-01
```

#### 2. 仅预热K线数据（不计算特征）

```bash
python scripts/prepare_training_data.py --symbols all --kline-only
```

**适用场景：**
- 只需要原始数据
- 特征计算在训练时进行
- 节省磁盘空间

#### 3. 仅预热特征数据（K线缓存已存在）

```bash
python scripts/prepare_training_data.py --symbols all --features-only
```

**适用场景：**
- K线数据已缓存
- 只需要更新特征
- 特征计算逻辑有变化

#### 4. 限制处理数量（测试用）

```bash
# 只处理前10只股票
python scripts/prepare_training_data.py --symbols all --limit 10
```

#### 5. 自定义缓存目录

```bash
python scripts/prepare_training_data.py \
    --symbols all \
    --kline-cache-dir /data/cache/kline \
    --feature-cache-dir /data/cache/features
```

## 预热后的使用

### 训练时使用缓存

```python
from src.data.cached_loader import ParquetDataLoader, FeatureCache

# 1. 使用K线缓存
loader = ParquetDataLoader(cache_dir="data/parquet")
df = loader.load_kline_data("000001", "2021-01-01", "2024-12-31", use_cache=True)
# 从缓存加载，速度快10-20倍！

# 2. 使用特征缓存
feature_cache = FeatureCache(cache_dir="data/features")
df_features = feature_cache.get_features("000001", "2021-01-01", "2024-12-31")
# 直接获取特征，无需重新计算！

if df_features is None:
    # 如果缓存不存在，计算特征
    df_features = builder.build_feature_matrix(df)
    feature_cache.save_features("000001", df_features)
```

### 批量训练示例

```python
from src.data.cached_loader import ParquetDataLoader, FeatureCache

# 初始化加载器
loader = ParquetDataLoader()
feature_cache = FeatureCache()

# 批量训练
for symbol in all_symbols:
    # 从缓存快速加载
    df_features = feature_cache.get_features(symbol, start_date, end_date)
    
    if df_features is not None:
        # 准备训练数据
        X, y = prepare_training_data(df_features)
        
        # 训练模型
        train_model(symbol, X, y)
```

## 性能对比

### 实际测试结果

**测试条件：**
- 股票数量：10只
- 数据范围：3年（~750条/只）
- 特征数量：45个

**结果对比：**

| 操作 | MySQL直接 | 使用缓存 | 提升 |
|------|----------|---------|------|
| 加载K线数据 | 5.2秒 | 0.4秒 | 13x ⚡ |
| 计算特征 | 18.5秒 | 0.8秒 | 23x ⚡⚡ |
| 总耗时 | 23.7秒 | 1.2秒 | 20x ⚡⚡ |

**扩展到100只股票：**
- MySQL直接：~240秒（4分钟）
- 使用缓存：~12秒
- 提升：20倍

**扩展到5000只股票：**
- MySQL直接：~12,000秒（3.3小时）
- 使用缓存：~600秒（10分钟）
- 提升：20倍

## 缓存管理

### 查看缓存信息

```python
from src.data.cached_loader import ParquetDataLoader, FeatureCache

# K线缓存信息
loader = ParquetDataLoader()
info = loader.get_cache_info()
print(f"缓存股票数: {info['cached_stocks']}")
print(f"缓存大小: {info['total_size_mb']:.2f} MB")
print(f"缓存有效期: {info['cache_ttl_hours']} 小时")

# 特征缓存信息
cache = FeatureCache()
info = cache.get_cache_info()
print(f"缓存股票数: {info['cached_stocks']}")
print(f"缓存大小: {info['total_size_mb']:.2f} MB")
```

### 清除缓存

```python
# 清除特定股票的缓存
loader.clear_cache(symbol="000001")

# 清除所有缓存
loader.clear_cache()

# 清除特征缓存
feature_cache.clear_cache()
```

### 更新缓存

```python
# 强制刷新缓存（重新从MySQL加载）
df = loader.load_kline_data(
    "000001",
    "2021-01-01",
    "2024-12-31",
    use_cache=True,
    force_refresh=True  # 强制刷新
)
```

## 缓存策略

### 缓存有效期

**默认：24小时**

```python
# 自定义缓存有效期
loader = ParquetDataLoader(
    cache_dir="data/parquet",
    cache_ttl=86400  # 24小时（秒）
)

# 1小时有效期
loader = ParquetDataLoader(cache_ttl=3600)

# 7天有效期
loader = ParquetDataLoader(cache_ttl=604800)
```

**工作原理：**
```python
# 检查缓存是否过期
cache_age = time.time() - cache_file.stat().st_mtime

if cache_age < cache_ttl:
    # 缓存有效，从缓存加载
    df = pd.read_parquet(cache_file)
else:
    # 缓存过期，从MySQL重新加载
    df = mysql_loader.load_kline_data(...)
```

### 何时需要重新预热

**需要重新预热的情况：**

1. **数据更新** - 新的交易日数据
   ```bash
   # 每天收盘后更新
   python scripts/prepare_training_data.py --symbols all
   ```

2. **特征逻辑变化** - 修改了特征计算方法
   ```bash
   # 清除旧特征，重新计算
   python scripts/prepare_training_data.py --symbols all --features-only
   ```

3. **新增股票** - 有新股上市
   ```bash
   # 只预热新股票
   python scripts/prepare_training_data.py --symbols 688XXX
   ```

4. **缓存过期** - 超过有效期
   ```bash
   # 自动检测，过期会重新加载
   ```

## 最佳实践

### 1. 首次使用前预热

```bash
# 第一次使用系统时
python scripts/prepare_training_data.py --symbols all --start-date 2021-01-01
```

### 2. 定期更新缓存

```bash
# 每天收盘后执行（可以设置定时任务）
0 16 * * 1-5 python scripts/prepare_training_data.py --symbols all
```

### 3. 按需预热

```bash
# 只预热需要训练的股票
python scripts/prepare_training_data.py --symbols 000001,600519,000002
```

### 4. 分阶段预热

```bash
# 先预热K线数据（快）
python scripts/prepare_training_data.py --symbols all --kline-only

# 后台预热特征数据（慢）
nohup python scripts/prepare_training_data.py --symbols all --features-only &
```

### 5. 监控磁盘空间

```python
# 定期检查缓存大小
loader = ParquetDataLoader()
info = loader.get_cache_info()

if info['total_size_mb'] > 50000:  # 超过50GB
    print("警告：缓存过大，考虑清理")
```

## 总结

### 数据预热的本质

**一句话总结：** 数据预热就是**提前把MySQL数据库中的数据复制到本地磁盘**，后续训练直接从本地读取，避免重复查询数据库。

### 核心优势

1. **速度快** - 本地磁盘读取比网络查询快10-20倍
2. **降低负载** - 减少数据库查询压力
3. **支持离线** - 预热后可以断开数据库连接
4. **节省时间** - 特别适合反复训练场景

### 适用场景

- ✅ 超参数调优（需要多次训练）
- ✅ 批量训练（训练多只股票）
- ✅ 模型对比（对比不同模型）
- ✅ 实验研究（频繁修改代码）
- ✅ 生产环境（降低数据库负载）

### 不适用场景

- ❌ 只训练一次（预热收益不明显）
- ❌ 实时数据（需要最新数据）
- ❌ 磁盘空间不足（缓存需要空间）

**推荐：** 对于大规模训练场景，数据预热是必备的性能优化手段！
