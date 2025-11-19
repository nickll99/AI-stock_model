"""测试特征缓存功能"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.cached_loader import ParquetDataLoader, FeatureCache
from src.features.dataset_builder import FeatureDatasetBuilder

def main():
    print("=" * 70)
    print("  测试特征缓存功能")
    print("=" * 70)
    
    symbol = "000001"
    
    # 1. 加载K线数据
    print(f"\n1. 加载K线数据: {symbol}")
    kline_loader = ParquetDataLoader(cache_dir="data/parquet")
    df = kline_loader.load_kline_data(symbol, "2021-01-01", "2024-12-31", use_cache=True)
    print(f"✓ K线数据: {len(df)} 条")
    
    # 2. 构建特征
    print(f"\n2. 构建特征...")
    builder = FeatureDatasetBuilder()
    df_features = builder.build_feature_matrix(df)
    print(f"✓ 特征数据: {len(df_features)} 条, {df_features.shape[1]} 列")
    
    # 3. 保存特征缓存
    print(f"\n3. 保存特征缓存...")
    feature_cache = FeatureCache(cache_dir="data/features")
    feature_cache.save_features(symbol, df_features)
    
    # 检查文件是否存在
    cache_file = Path(f"data/features/{symbol}_features.parquet")
    if cache_file.exists():
        size_mb = cache_file.stat().st_size / 1024 / 1024
        print(f"✓ 缓存文件已创建: {cache_file}")
        print(f"✓ 文件大小: {size_mb:.2f} MB")
    else:
        print(f"✗ 缓存文件不存在: {cache_file}")
        return
    
    # 4. 加载特征缓存
    print(f"\n4. 加载特征缓存...")
    df_loaded = feature_cache.load_features(symbol)
    
    if df_loaded is not None:
        print(f"✓ 从缓存加载: {len(df_loaded)} 条, {df_loaded.shape[1]} 列")
        
        # 验证数据一致性
        if len(df_loaded) == len(df_features) and df_loaded.shape[1] == df_features.shape[1]:
            print(f"✓ 数据一致性验证通过")
        else:
            print(f"✗ 数据不一致！")
            print(f"  原始: {df_features.shape}")
            print(f"  加载: {df_loaded.shape}")
    else:
        print(f"✗ 加载失败")
    
    # 5. 查看缓存信息
    print(f"\n5. 缓存信息...")
    cache_info = feature_cache.get_cache_info()
    print(f"✓ 缓存目录: {cache_info['cache_dir']}")
    print(f"✓ 缓存股票数: {cache_info['cached_stocks']}")
    print(f"✓ 缓存大小: {cache_info['total_size_mb']:.2f} MB")
    
    print("\n" + "=" * 70)
    print("  测试完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
