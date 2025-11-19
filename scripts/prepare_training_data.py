"""
数据预热脚本 - 提前加载和缓存训练数据

使用方法:
    python scripts/prepare_training_data.py --symbols all --start-date 2021-01-01 --end-date 2024-12-31
    python scripts/prepare_training_data.py --symbols 000001,600519 --start-date 2021-01-01
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime, timedelta
from typing import List

from src.data.loader import StockDataLoader
from src.data.cached_loader import ParquetDataLoader, FeatureCache
from src.features.dataset_builder import FeatureDatasetBuilder


def prepare_kline_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    cache_dir: str = "data/parquet"
):
    """
    预热K线数据
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        cache_dir: 缓存目录
    """
    print("=" * 70)
    print("  预热K线数据")
    print("=" * 70)
    print(f"\n股票数量: {len(symbols)}")
    print(f"日期范围: {start_date} 至 {end_date}")
    print(f"缓存目录: {cache_dir}\n")
    
    loader = ParquetDataLoader(cache_dir=cache_dir)
    
    success_count = 0
    fail_count = 0
    total_records = 0
    
    for i, symbol in enumerate(symbols, 1):
        try:
            # 加载数据（会自动缓存）
            df = loader.load_kline_data(
                symbol,
                start_date,
                end_date,
                use_cache=True,
                force_refresh=True  # 强制刷新
            )
            
            if df.empty:
                print(f"[{i}/{len(symbols)}] {symbol}: 无数据")
                fail_count += 1
            else:
                print(f"[{i}/{len(symbols)}] {symbol}: {len(df)} 条记录已缓存")
                success_count += 1
                total_records += len(df)
            
        except Exception as e:
            print(f"[{i}/{len(symbols)}] {symbol}: 失败 - {e}")
            fail_count += 1
    
    print(f"\n" + "=" * 70)
    print(f"K线数据预热完成")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  总记录数: {total_records:,}")
    
    # 显示缓存信息
    cache_info = loader.get_cache_info()
    print(f"  缓存大小: {cache_info['total_size_mb']:.2f} MB")
    print("=" * 70)


def prepare_features(
    symbols: List[str],
    start_date: str,
    end_date: str,
    kline_cache_dir: str = "data/parquet",
    feature_cache_dir: str = "data/features"
):
    """
    预热特征数据
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        kline_cache_dir: K线缓存目录
        feature_cache_dir: 特征缓存目录
    """
    print("\n" + "=" * 70)
    print("  预热特征数据")
    print("=" * 70)
    print(f"\n股票数量: {len(symbols)}")
    print(f"特征缓存目录: {feature_cache_dir}\n")
    
    kline_loader = ParquetDataLoader(cache_dir=kline_cache_dir)
    feature_cache = FeatureCache(cache_dir=feature_cache_dir)
    builder = FeatureDatasetBuilder()
    
    success_count = 0
    fail_count = 0
    
    for i, symbol in enumerate(symbols, 1):
        try:
            # 加载K线数据
            df = kline_loader.load_kline_data(symbol, start_date, end_date)
            
            if df.empty:
                print(f"[{i}/{len(symbols)}] {symbol}: 无数据")
                fail_count += 1
                continue
            
            # 计算特征
            df_features = builder.build_feature_matrix(df)
            
            # 保存特征缓存
            feature_cache.save_features(symbol, df_features)
            
            print(f"[{i}/{len(symbols)}] {symbol}: {len(df_features)} 条记录, {df_features.shape[1]} 个特征已缓存")
            success_count += 1
            
        except Exception as e:
            print(f"[{i}/{len(symbols)}] {symbol}: 失败 - {e}")
            fail_count += 1
    
    print(f"\n" + "=" * 70)
    print(f"特征数据预热完成")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    
    # 显示缓存信息
    cache_info = feature_cache.get_cache_info()
    print(f"  缓存大小: {cache_info['total_size_mb']:.2f} MB")
    print("=" * 70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据预热脚本")
    
    parser.add_argument(
        "--symbols",
        type=str,
        default="all",
        help="股票代码，用逗号分隔，或使用'all'表示所有活跃股票"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d"),
        help="开始日期 (YYYY-MM-DD)，默认3年前"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="结束日期 (YYYY-MM-DD)，默认今天"
    )
    
    parser.add_argument(
        "--kline-only",
        action="store_true",
        help="仅预热K线数据，不计算特征"
    )
    
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="仅预热特征数据（需要K线缓存已存在）"
    )
    
    parser.add_argument(
        "--kline-cache-dir",
        type=str,
        default="data/parquet",
        help="K线缓存目录"
    )
    
    parser.add_argument(
        "--feature-cache-dir",
        type=str,
        default="data/features",
        help="特征缓存目录"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理的股票数量（用于测试）"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  数据预热脚本")
    print("=" * 70)
    
    # 获取股票列表
    if args.symbols == "all":
        print("\n获取所有活跃股票...")
        loader = StockDataLoader()
        symbols = loader.get_all_active_stocks()
        print(f"找到 {len(symbols)} 只活跃股票")
    else:
        symbols = [s.strip() for s in args.symbols.split(",")]
        print(f"\n指定股票: {', '.join(symbols)}")
    
    # 限制数量（用于测试）
    if args.limit:
        symbols = symbols[:args.limit]
        print(f"限制处理数量: {args.limit}")
    
    # 预热K线数据
    if not args.features_only:
        prepare_kline_data(
            symbols,
            args.start_date,
            args.end_date,
            args.kline_cache_dir
        )
    
    # 预热特征数据
    if not args.kline_only:
        prepare_features(
            symbols,
            args.start_date,
            args.end_date,
            args.kline_cache_dir,
            args.feature_cache_dir
        )
    
    print("\n" + "=" * 70)
    print("  数据预热完成！")
    print("=" * 70)
    print("\n下一步:")
    print("  1. 使用缓存进行训练")
    print("  2. 训练速度将大幅提升")
    print("  3. 数据库负载将大幅降低")
    print()


if __name__ == "__main__":
    main()
