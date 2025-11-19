"""
数据加载性能基准测试

比较不同加载方式的性能
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from datetime import datetime, timedelta

from src.data.loader import StockDataLoader
from src.data.cached_loader import ParquetDataLoader, FeatureCache
from src.features.dataset_builder import FeatureDatasetBuilder


def benchmark_mysql_loading(symbols, start_date, end_date):
    """测试MySQL直接加载"""
    print("\n" + "=" * 70)
    print("  测试1: MySQL直接加载")
    print("=" * 70)
    
    loader = StockDataLoader()
    
    start_time = time.time()
    total_records = 0
    
    for symbol in symbols:
        df = loader.load_kline_data(symbol, start_date, end_date)
        total_records += len(df)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n结果:")
    print(f"  股票数量: {len(symbols)}")
    print(f"  总记录数: {total_records:,}")
    print(f"  总耗时: {elapsed_time:.2f}秒")
    print(f"  平均每只: {elapsed_time/len(symbols):.3f}秒")
    print(f"  吞吐量: {total_records/elapsed_time:.0f} 记录/秒")
    
    return elapsed_time


def benchmark_parquet_loading_first(symbols, start_date, end_date):
    """测试Parquet首次加载（需要从MySQL加载并缓存）"""
    print("\n" + "=" * 70)
    print("  测试2: Parquet首次加载（含缓存写入）")
    print("=" * 70)
    
    # 清除缓存
    loader = ParquetDataLoader(cache_dir="data/benchmark/parquet")
    loader.clear_cache()
    
    start_time = time.time()
    total_records = 0
    
    for symbol in symbols:
        df = loader.load_kline_data(symbol, start_date, end_date, use_cache=True)
        total_records += len(df)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n结果:")
    print(f"  股票数量: {len(symbols)}")
    print(f"  总记录数: {total_records:,}")
    print(f"  总耗时: {elapsed_time:.2f}秒")
    print(f"  平均每只: {elapsed_time/len(symbols):.3f}秒")
    print(f"  吞吐量: {total_records/elapsed_time:.0f} 记录/秒")
    
    cache_info = loader.get_cache_info()
    print(f"  缓存大小: {cache_info['total_size_mb']:.2f} MB")
    
    return elapsed_time


def benchmark_parquet_loading_cached(symbols, start_date, end_date):
    """测试Parquet缓存加载"""
    print("\n" + "=" * 70)
    print("  测试3: Parquet缓存加载")
    print("=" * 70)
    
    loader = ParquetDataLoader(cache_dir="data/benchmark/parquet")
    
    start_time = time.time()
    total_records = 0
    
    for symbol in symbols:
        df = loader.load_kline_data(symbol, start_date, end_date, use_cache=True)
        total_records += len(df)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n结果:")
    print(f"  股票数量: {len(symbols)}")
    print(f"  总记录数: {total_records:,}")
    print(f"  总耗时: {elapsed_time:.2f}秒")
    print(f"  平均每只: {elapsed_time/len(symbols):.3f}秒")
    print(f"  吞吐量: {total_records/elapsed_time:.0f} 记录/秒")
    
    return elapsed_time


def benchmark_feature_calculation(symbols, start_date, end_date):
    """测试特征计算"""
    print("\n" + "=" * 70)
    print("  测试4: 特征计算（含技术指标）")
    print("=" * 70)
    
    loader = ParquetDataLoader(cache_dir="data/benchmark/parquet")
    builder = FeatureDatasetBuilder()
    
    start_time = time.time()
    total_records = 0
    total_features = 0
    
    for symbol in symbols:
        df = loader.load_kline_data(symbol, start_date, end_date, use_cache=True)
        if not df.empty:
            df_features = builder.build_feature_matrix(df)
            total_records += len(df_features)
            total_features = df_features.shape[1]
    
    elapsed_time = time.time() - start_time
    
    print(f"\n结果:")
    print(f"  股票数量: {len(symbols)}")
    print(f"  总记录数: {total_records:,}")
    print(f"  特征数量: {total_features}")
    print(f"  总耗时: {elapsed_time:.2f}秒")
    print(f"  平均每只: {elapsed_time/len(symbols):.3f}秒")
    print(f"  吞吐量: {total_records/elapsed_time:.0f} 记录/秒")
    
    return elapsed_time


def benchmark_feature_cache(symbols, start_date, end_date):
    """测试特征缓存"""
    print("\n" + "=" * 70)
    print("  测试5: 特征缓存（首次）")
    print("=" * 70)
    
    loader = ParquetDataLoader(cache_dir="data/benchmark/parquet")
    feature_cache = FeatureCache(cache_dir="data/benchmark/features")
    feature_cache.clear_cache()
    builder = FeatureDatasetBuilder()
    
    start_time = time.time()
    total_records = 0
    
    for symbol in symbols:
        df = loader.load_kline_data(symbol, start_date, end_date, use_cache=True)
        if not df.empty:
            df_features = builder.build_feature_matrix(df)
            feature_cache.save_features(symbol, df_features)
            total_records += len(df_features)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n结果:")
    print(f"  股票数量: {len(symbols)}")
    print(f"  总记录数: {total_records:,}")
    print(f"  总耗时: {elapsed_time:.2f}秒")
    print(f"  平均每只: {elapsed_time/len(symbols):.3f}秒")
    
    cache_info = feature_cache.get_cache_info()
    print(f"  缓存大小: {cache_info['total_size_mb']:.2f} MB")
    
    return elapsed_time


def benchmark_feature_cache_loading(symbols, start_date, end_date):
    """测试特征缓存加载"""
    print("\n" + "=" * 70)
    print("  测试6: 特征缓存加载")
    print("=" * 70)
    
    feature_cache = FeatureCache(cache_dir="data/benchmark/features")
    
    start_time = time.time()
    total_records = 0
    
    for symbol in symbols:
        df_features = feature_cache.get_features(symbol, start_date, end_date, use_cache=True)
        if df_features is not None:
            total_records += len(df_features)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n结果:")
    print(f"  股票数量: {len(symbols)}")
    print(f"  总记录数: {total_records:,}")
    print(f"  总耗时: {elapsed_time:.2f}秒")
    print(f"  平均每只: {elapsed_time/len(symbols):.3f}秒")
    print(f"  吞吐量: {total_records/elapsed_time:.0f} 记录/秒")
    
    return elapsed_time


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("  数据加载性能基准测试")
    print("=" * 70)
    
    # 测试参数
    num_stocks = 10  # 测试股票数量
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
    
    print(f"\n测试配置:")
    print(f"  股票数量: {num_stocks}")
    print(f"  日期范围: {start_date} 至 {end_date}")
    print(f"  预计数据量: ~{num_stocks * 750:,} 条记录")
    
    # 获取测试股票
    print(f"\n获取测试股票...")
    loader = StockDataLoader()
    all_symbols = loader.get_all_active_stocks()
    test_symbols = all_symbols[:num_stocks]
    
    print(f"测试股票: {', '.join(test_symbols)}")
    
    # 运行基准测试
    results = {}
    
    try:
        # 测试1: MySQL直接加载
        results['mysql'] = benchmark_mysql_loading(test_symbols, start_date, end_date)
        
        # 测试2: Parquet首次加载
        results['parquet_first'] = benchmark_parquet_loading_first(test_symbols, start_date, end_date)
        
        # 测试3: Parquet缓存加载
        results['parquet_cached'] = benchmark_parquet_loading_cached(test_symbols, start_date, end_date)
        
        # 测试4: 特征计算
        results['feature_calc'] = benchmark_feature_calculation(test_symbols, start_date, end_date)
        
        # 测试5: 特征缓存（首次）
        results['feature_cache_first'] = benchmark_feature_cache(test_symbols, start_date, end_date)
        
        # 测试6: 特征缓存加载
        results['feature_cache_load'] = benchmark_feature_cache_loading(test_symbols, start_date, end_date)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 输出对比结果
    print("\n" + "=" * 70)
    print("  性能对比总结")
    print("=" * 70)
    
    print(f"\n加载方式对比:")
    print(f"  MySQL直接加载:        {results['mysql']:.2f}秒 (基准)")
    print(f"  Parquet首次加载:      {results['parquet_first']:.2f}秒 ({results['parquet_first']/results['mysql']:.2f}x)")
    print(f"  Parquet缓存加载:      {results['parquet_cached']:.2f}秒 ({results['parquet_cached']/results['mysql']:.2f}x) ⚡")
    
    print(f"\n特征处理对比:")
    print(f"  特征计算:             {results['feature_calc']:.2f}秒")
    print(f"  特征缓存（首次）:     {results['feature_cache_first']:.2f}秒")
    print(f"  特征缓存加载:         {results['feature_cache_load']:.2f}秒 ⚡⚡")
    
    print(f"\n性能提升:")
    speedup_parquet = results['mysql'] / results['parquet_cached']
    speedup_feature = results['feature_calc'] / results['feature_cache_load']
    
    print(f"  Parquet缓存:          {speedup_parquet:.1f}x 提升")
    print(f"  特征缓存:             {speedup_feature:.1f}x 提升")
    
    print(f"\n推荐方案:")
    print(f"  ✓ 使用Parquet缓存K线数据")
    print(f"  ✓ 使用特征缓存避免重复计算")
    print(f"  ✓ 首次训练后，后续训练速度提升 {speedup_feature:.0f}倍以上")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
