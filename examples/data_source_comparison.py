"""
数据源对比示例
演示如何使用不同的数据源
"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import StockDataLoader
from src.data.cached_loader import ParquetDataLoader


def example_mysql_direct():
    """示例1: 直接从MySQL加载"""
    print("=" * 70)
    print("  示例1: 直接从MySQL加载（不使用缓存）")
    print("=" * 70)
    
    loader = StockDataLoader()
    
    print("\n使用 StockDataLoader - 总是从MySQL读取")
    print("适用场景: 需要最新数据、实时预测\n")
    
    start_time = time.time()
    df = loader.load_kline_data("000001", "2023-01-01", "2024-12-31")
    elapsed_time = time.time() - start_time
    
    print(f"✓ 加载完成")
    print(f"  数据来源: MySQL数据库")
    print(f"  记录数: {len(df)}")
    print(f"  耗时: {elapsed_time:.3f}秒")
    print(f"  特点: 总是最新数据，但速度较慢")


def example_cache_auto():
    """示例2: 自动使用缓存"""
    print("\n" + "=" * 70)
    print("  示例2: 自动使用缓存（智能模式）")
    print("=" * 70)
    
    loader = ParquetDataLoader(cache_dir="data/parquet")
    
    print("\n使用 ParquetDataLoader - 自动判断使用缓存或MySQL")
    print("适用场景: 大多数训练场景\n")
    
    # 第一次加载
    print("第一次加载:")
    start_time = time.time()
    df1 = loader.load_kline_data("000001", "2023-01-01", "2024-12-31", use_cache=True)
    elapsed_time1 = time.time() - start_time
    
    print(f"✓ 加载完成")
    print(f"  记录数: {len(df1)}")
    print(f"  耗时: {elapsed_time1:.3f}秒")
    print(f"  说明: 缓存不存在，从MySQL加载并创建缓存")
    
    # 第二次加载
    print("\n第二次加载:")
    start_time = time.time()
    df2 = loader.load_kline_data("000001", "2023-01-01", "2024-12-31", use_cache=True)
    elapsed_time2 = time.time() - start_time
    
    print(f"✓ 加载完成")
    print(f"  记录数: {len(df2)}")
    print(f"  耗时: {elapsed_time2:.3f}秒")
    print(f"  说明: 从缓存加载，速度快{elapsed_time1/elapsed_time2:.1f}倍！")


def example_cache_disabled():
    """示例3: 禁用缓存"""
    print("\n" + "=" * 70)
    print("  示例3: 禁用缓存（强制使用MySQL）")
    print("=" * 70)
    
    loader = ParquetDataLoader(cache_dir="data/parquet")
    
    print("\n使用 ParquetDataLoader 但禁用缓存")
    print("适用场景: 需要最新数据但想保留缓存功能\n")
    
    start_time = time.time()
    df = loader.load_kline_data(
        "000001",
        "2023-01-01",
        "2024-12-31",
        use_cache=False  # 禁用缓存
    )
    elapsed_time = time.time() - start_time
    
    print(f"✓ 加载完成")
    print(f"  数据来源: MySQL数据库（忽略缓存）")
    print(f"  记录数: {len(df)}")
    print(f"  耗时: {elapsed_time:.3f}秒")
    print(f"  特点: 不读取也不写入缓存")


def example_force_refresh():
    """示例4: 强制刷新缓存"""
    print("\n" + "=" * 70)
    print("  示例4: 强制刷新缓存")
    print("=" * 70)
    
    loader = ParquetDataLoader(cache_dir="data/parquet")
    
    print("\n强制从MySQL重新加载并更新缓存")
    print("适用场景: 数据更新后需要刷新缓存\n")
    
    start_time = time.time()
    df = loader.load_kline_data(
        "000001",
        "2023-01-01",
        "2024-12-31",
        use_cache=True,
        force_refresh=True  # 强制刷新
    )
    elapsed_time = time.time() - start_time
    
    print(f"✓ 加载完成")
    print(f"  数据来源: MySQL数据库")
    print(f"  记录数: {len(df)}")
    print(f"  耗时: {elapsed_time:.3f}秒")
    print(f"  特点: 忽略现有缓存，重新加载并更新缓存")


def example_mixed_mode():
    """示例5: 混合模式"""
    print("\n" + "=" * 70)
    print("  示例5: 混合模式（根据场景选择）")
    print("=" * 70)
    
    print("\n根据不同场景使用不同的数据源\n")
    
    # 训练时使用缓存
    print("场景1: 模型训练 - 使用缓存")
    cache_loader = ParquetDataLoader()
    start_time = time.time()
    df_train = cache_loader.load_kline_data("000001", "2021-01-01", "2023-12-31", use_cache=True)
    elapsed_time = time.time() - start_time
    print(f"  ✓ 训练数据: {len(df_train)} 条, 耗时 {elapsed_time:.3f}秒")
    
    # 实时预测使用MySQL
    print("\n场景2: 实时预测 - 使用MySQL")
    mysql_loader = StockDataLoader()
    start_time = time.time()
    df_predict = mysql_loader.load_kline_data("000001", "2024-12-01", "2024-12-31")
    elapsed_time = time.time() - start_time
    print(f"  ✓ 预测数据: {len(df_predict)} 条, 耗时 {elapsed_time:.3f}秒")
    
    print("\n  说明: 训练用缓存提升速度，预测用MySQL确保最新")


def example_check_cache():
    """示例6: 检查缓存状态"""
    print("\n" + "=" * 70)
    print("  示例6: 检查缓存状态")
    print("=" * 70)
    
    loader = ParquetDataLoader(cache_dir="data/parquet")
    
    print("\n查看缓存信息:\n")
    
    cache_info = loader.get_cache_info()
    
    print(f"  缓存目录: {cache_info['cache_dir']}")
    print(f"  缓存股票数: {cache_info['cached_stocks']}")
    print(f"  缓存大小: {cache_info['total_size_mb']:.2f} MB")
    print(f"  缓存有效期: {cache_info['cache_ttl_hours']:.1f} 小时")
    
    # 检查特定股票的缓存
    cache_file = Path("data/parquet/000001.parquet")
    if cache_file.exists():
        import time
        cache_age = time.time() - cache_file.stat().st_mtime
        cache_age_hours = cache_age / 3600
        
        print(f"\n  000001 缓存状态:")
        print(f"    文件: {cache_file}")
        print(f"    大小: {cache_file.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"    缓存时间: {cache_age_hours:.1f} 小时前")
        print(f"    状态: {'有效' if cache_age < cache_info['cache_ttl_hours'] * 3600 else '已过期'}")
    else:
        print(f"\n  000001 缓存不存在")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("  数据源对比示例")
    print("  演示如何区分和控制数据来源")
    print("=" * 70)
    
    try:
        # 示例1: 直接MySQL
        example_mysql_direct()
        
        # 示例2: 自动缓存
        example_cache_auto()
        
        # 示例3: 禁用缓存
        example_cache_disabled()
        
        # 示例4: 强制刷新
        example_force_refresh()
        
        # 示例5: 混合模式
        example_mixed_mode()
        
        # 示例6: 检查缓存
        example_check_cache()
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 总结
    print("\n" + "=" * 70)
    print("  总结")
    print("=" * 70)
    
    print("\n数据源选择指南:")
    print("  1. StockDataLoader()           → 总是使用MySQL")
    print("  2. ParquetDataLoader()          → 智能使用缓存")
    print("  3. use_cache=False              → 禁用缓存")
    print("  4. force_refresh=True           → 强制刷新缓存")
    print("  5. 混合模式                     → 根据场景选择")
    
    print("\n推荐使用:")
    print("  ✓ 开发调试: ParquetDataLoader() - 使用缓存")
    print("  ✓ 生产训练: ParquetDataLoader() - 使用缓存")
    print("  ✓ 实时预测: StockDataLoader()   - 直接MySQL")
    print("  ✓ 数据验证: force_refresh=True  - 强制刷新")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
