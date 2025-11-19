"""
测试训练脚本是否正确使用缓存数据

使用方法:
    python test_cache_usage.py
"""
import os
import sys
from pathlib import Path

def check_cache_directories():
    """检查缓存目录"""
    print("\n" + "="*70)
    print("  检查缓存目录")
    print("="*70)
    
    kline_dir = Path("data/parquet")
    feature_dir = Path("data/features")
    
    # 检查K线缓存
    if kline_dir.exists():
        kline_files = list(kline_dir.glob("*.parquet"))
        print(f"\n✅ K线缓存目录存在: {kline_dir}")
        print(f"   文件数量: {len(kline_files)}")
        if kline_files:
            total_size = sum(f.stat().st_size for f in kline_files)
            print(f"   总大小: {total_size / 1024 / 1024:.2f} MB")
    else:
        print(f"\n❌ K线缓存目录不存在: {kline_dir}")
        print(f"   请先运行: python scripts/prepare_training_data.py --symbols all --workers 8 --resume")
    
    # 检查特征缓存
    if feature_dir.exists():
        feature_files = list(feature_dir.glob("*_features.parquet"))
        print(f"\n✅ 特征缓存目录存在: {feature_dir}")
        print(f"   文件数量: {len(feature_files)}")
        if feature_files:
            total_size = sum(f.stat().st_size for f in feature_files)
            print(f"   总大小: {total_size / 1024 / 1024:.2f} MB")
    else:
        print(f"\n❌ 特征缓存目录不存在: {feature_dir}")
        print(f"   请先运行: python scripts/prepare_training_data.py --symbols all --workers 8 --resume")
    
    return kline_dir.exists() and feature_dir.exists()


def test_cache_loading():
    """测试缓存加载"""
    print("\n" + "="*70)
    print("  测试缓存加载")
    print("="*70)
    
    try:
        from src.data.cached_loader import ParquetDataLoader, FeatureCache
        
        # 测试K线缓存
        print("\n测试K线缓存加载...")
        kline_loader = ParquetDataLoader(cache_dir="data/parquet")
        
        # 尝试加载一只股票
        test_symbol = "000001"
        df = kline_loader.load_kline_data(test_symbol, "2024-01-01", "2024-12-31")
        
        if df is not None and len(df) > 0:
            print(f"✅ 成功从缓存加载 {test_symbol}")
            print(f"   数据行数: {len(df)}")
            print(f"   数据列: {list(df.columns)}")
        else:
            print(f"❌ 缓存中没有 {test_symbol} 的数据")
        
        # 测试特征缓存
        print("\n测试特征缓存加载...")
        feature_cache = FeatureCache(cache_dir="data/features")
        
        df_features = feature_cache.load_features(test_symbol)
        
        if df_features is not None and len(df_features) > 0:
            print(f"✅ 成功从缓存加载 {test_symbol} 的特征")
            print(f"   特征行数: {len(df_features)}")
            print(f"   特征数量: {len(df_features.columns)}")
        else:
            print(f"❌ 缓存中没有 {test_symbol} 的特征数据")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 缓存加载测试失败: {e}")
        return False


def check_training_script():
    """检查训练脚本配置"""
    print("\n" + "="*70)
    print("  检查训练脚本配置")
    print("="*70)
    
    script_path = Path("scripts/train_universal_model.py")
    
    if not script_path.exists():
        print(f"\n❌ 训练脚本不存在: {script_path}")
        return False
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否有缓存相关代码
    checks = {
        "ParquetDataLoader导入": "from src.data.cached_loader import ParquetDataLoader",
        "FeatureCache导入": "FeatureCache",
        "use_cache参数": "use_cache",
        "kline_cache_dir参数": "kline-cache-dir",
        "feature_cache_dir参数": "feature-cache-dir",
    }
    
    print("\n检查项:")
    all_passed = True
    for name, pattern in checks.items():
        if pattern in content:
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")
            all_passed = False
    
    return all_passed


def main():
    """主函数"""
    print("\n" + "="*70)
    print("  缓存使用测试")
    print("="*70)
    
    # 检查缓存目录
    cache_ok = check_cache_directories()
    
    # 测试缓存加载
    if cache_ok:
        loading_ok = test_cache_loading()
    else:
        loading_ok = False
        print("\n⚠️  跳过缓存加载测试（缓存目录不存在）")
    
    # 检查训练脚本
    script_ok = check_training_script()
    
    # 总结
    print("\n" + "="*70)
    print("  测试总结")
    print("="*70)
    
    if cache_ok and loading_ok and script_ok:
        print("\n✅ 所有测试通过！")
        print("\n你可以直接运行训练脚本，它会自动使用缓存数据：")
        print("  python scripts/train_universal_model.py")
    else:
        print("\n❌ 部分测试失败")
        
        if not cache_ok:
            print("\n⚠️  缓存目录不存在或为空")
            print("   请先运行数据预热：")
            print("   python scripts/prepare_training_data.py --symbols all --workers 8 --resume")
        
        if not loading_ok:
            print("\n⚠️  缓存加载失败")
            print("   请检查缓存文件是否完整")
        
        if not script_ok:
            print("\n⚠️  训练脚本配置有问题")
            print("   请检查脚本是否正确导入了缓存加载器")
    
    print()


if __name__ == "__main__":
    main()
