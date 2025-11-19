"""
数据预热脚本 - 提前加载和缓存训练数据

使用方法:
    # 基本用法
    python scripts/prepare_training_data.py --symbols all --start-date 2021-01-01 --end-date 2024-12-31
    
    # 多并发（4个进程）
    python scripts/prepare_training_data.py --symbols all --workers 4
    
    # 断点续传
    python scripts/prepare_training_data.py --symbols all --resume
    
    # 多并发 + 断点续传
    python scripts/prepare_training_data.py --symbols all --workers 124 --resume
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime, timedelta
from typing import List
from multiprocessing import Pool, cpu_count
from functools import partial

from src.data.loader import StockDataLoader
from src.data.cached_loader import ParquetDataLoader, FeatureCache
from src.features.dataset_builder import FeatureDatasetBuilder


def process_single_kline(args):
    """
    处理单只股票的K线数据（用于多进程）
    
    Args:
        args: (symbol, start_date, end_date, cache_dir, index, total)
        
    Returns:
        (symbol, success, record_count, error_msg)
    """
    symbol, start_date, end_date, cache_dir, index, total = args
    
    try:
        loader = ParquetDataLoader(cache_dir=cache_dir)
        df = loader.load_kline_data(
            symbol,
            start_date,
            end_date,
            use_cache=True,
            force_refresh=True
        )
        
        if df.empty:
            return (symbol, False, 0, "无数据")
        else:
            return (symbol, True, len(df), None)
            
    except Exception as e:
        return (symbol, False, 0, str(e))


def prepare_kline_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    cache_dir: str = "data/parquet",
    workers: int = 1,
    resume: bool = False,
    progress_file: str = "data/.progress_kline.json"
):
    """
    预热K线数据（支持多进程和断点续传）
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        cache_dir: 缓存目录
        workers: 并发进程数
        resume: 是否断点续传
        progress_file: 进度文件路径
    """
    print("=" * 70)
    print("  预热K线数据")
    print("=" * 70)
    print(f"\n股票数量: {len(symbols)}")
    print(f"日期范围: {start_date} 至 {end_date}")
    print(f"缓存目录: {cache_dir}")
    print(f"并发进程: {workers}")
    print(f"断点续传: {'是' if resume else '否'}\n")
    
    # 加载进度
    completed_symbols = set()
    if resume and Path(progress_file).exists():
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                completed_symbols = set(progress.get('completed', []))
            print(f"从断点恢复: 已完成 {len(completed_symbols)} 只股票\n")
        except Exception as e:
            print(f"加载进度文件失败: {e}\n")
    
    # 过滤已完成的股票
    remaining_symbols = [s for s in symbols if s not in completed_symbols]
    
    if not remaining_symbols:
        print("所有股票已完成！")
        return
    
    print(f"待处理股票: {len(remaining_symbols)}\n")
    
    success_count = len(completed_symbols)
    fail_count = 0
    total_records = 0
    
    # 准备参数
    total = len(symbols)
    args_list = [
        (symbol, start_date, end_date, cache_dir, i + len(completed_symbols) + 1, total)
        for i, symbol in enumerate(remaining_symbols)
    ]
    
    # 多进程处理
    if workers > 1:
        with Pool(workers) as pool:
            for result in pool.imap_unordered(process_single_kline, args_list):
                symbol, success, record_count, error_msg = result
                
                if success:
                    print(f"[{success_count + fail_count + 1}/{total}] {symbol}: {record_count} 条记录已缓存")
                    success_count += 1
                    total_records += record_count
                    completed_symbols.add(symbol)
                else:
                    print(f"[{success_count + fail_count + 1}/{total}] {symbol}: 失败 - {error_msg}")
                    fail_count += 1
                
                # 保存进度
                if resume and (success_count + fail_count) % 10 == 0:
                    save_progress(progress_file, list(completed_symbols))
    else:
        # 单进程处理
        for args in args_list:
            symbol, success, record_count, error_msg = process_single_kline(args)
            
            if success:
                print(f"[{success_count + fail_count + 1}/{total}] {symbol}: {record_count} 条记录已缓存")
                success_count += 1
                total_records += record_count
                completed_symbols.add(symbol)
            else:
                print(f"[{success_count + fail_count + 1}/{total}] {symbol}: 失败 - {error_msg}")
                fail_count += 1
            
            # 保存进度
            if resume and (success_count + fail_count) % 10 == 0:
                save_progress(progress_file, list(completed_symbols))
    
    # 保存最终进度
    if resume:
        save_progress(progress_file, list(completed_symbols))
    
    print(f"\n" + "=" * 70)
    print(f"K线数据预热完成")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  总记录数: {total_records:,}")
    
    # 显示缓存信息
    loader = ParquetDataLoader(cache_dir=cache_dir)
    cache_info = loader.get_cache_info()
    print(f"  缓存大小: {cache_info['total_size_mb']:.2f} MB")
    print("=" * 70)


def save_progress(progress_file: str, completed: List[str]):
    """保存进度"""
    Path(progress_file).parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump({
            'completed': completed,
            'timestamp': datetime.now().isoformat()
        }, f)


def process_single_feature(args):
    """
    处理单只股票的特征数据（用于多进程）
    
    Args:
        args: (symbol, start_date, end_date, kline_cache_dir, feature_cache_dir, index, total)
        
    Returns:
        (symbol, success, record_count, feature_count, error_msg)
    """
    symbol, start_date, end_date, kline_cache_dir, feature_cache_dir, index, total = args
    
    try:
        kline_loader = ParquetDataLoader(cache_dir=kline_cache_dir)
        feature_cache = FeatureCache(cache_dir=feature_cache_dir)
        builder = FeatureDatasetBuilder()
        
        # 加载K线数据
        df = kline_loader.load_kline_data(symbol, start_date, end_date)
        
        if df.empty:
            return (symbol, False, 0, 0, "无数据")
        
        # 计算特征
        df_features = builder.build_feature_matrix(df)
        
        # 保存特征缓存
        feature_cache.save_features(symbol, df_features)
        
        return (symbol, True, len(df_features), df_features.shape[1], None)
        
    except Exception as e:
        return (symbol, False, 0, 0, str(e))


def prepare_features(
    symbols: List[str],
    start_date: str,
    end_date: str,
    kline_cache_dir: str = "data/parquet",
    feature_cache_dir: str = "data/features",
    workers: int = 1,
    resume: bool = False,
    progress_file: str = "data/.progress_features.json"
):
    """
    预热特征数据（支持多进程和断点续传）
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        kline_cache_dir: K线缓存目录
        feature_cache_dir: 特征缓存目录
        workers: 并发进程数
        resume: 是否断点续传
        progress_file: 进度文件路径
    """
    print("\n" + "=" * 70)
    print("  预热特征数据")
    print("=" * 70)
    print(f"\n股票数量: {len(symbols)}")
    print(f"特征缓存目录: {feature_cache_dir}")
    print(f"并发进程: {workers}")
    print(f"断点续传: {'是' if resume else '否'}\n")
    
    # 加载进度
    completed_symbols = set()
    if resume and Path(progress_file).exists():
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                completed_symbols = set(progress.get('completed', []))
            print(f"从断点恢复: 已完成 {len(completed_symbols)} 只股票\n")
        except Exception as e:
            print(f"加载进度文件失败: {e}\n")
    
    # 过滤已完成的股票
    remaining_symbols = [s for s in symbols if s not in completed_symbols]
    
    if not remaining_symbols:
        print("所有股票已完成！")
        return
    
    print(f"待处理股票: {len(remaining_symbols)}\n")
    
    success_count = len(completed_symbols)
    fail_count = 0
    
    # 准备参数
    total = len(symbols)
    args_list = [
        (symbol, start_date, end_date, kline_cache_dir, feature_cache_dir, 
         i + len(completed_symbols) + 1, total)
        for i, symbol in enumerate(remaining_symbols)
    ]
    
    # 多进程处理
    if workers > 1:
        with Pool(workers) as pool:
            for result in pool.imap_unordered(process_single_feature, args_list):
                symbol, success, record_count, feature_count, error_msg = result
                
                if success:
                    print(f"[{success_count + fail_count + 1}/{total}] {symbol}: "
                          f"{record_count} 条记录, {feature_count} 个特征已缓存")
                    success_count += 1
                    completed_symbols.add(symbol)
                else:
                    print(f"[{success_count + fail_count + 1}/{total}] {symbol}: 失败 - {error_msg}")
                    fail_count += 1
                
                # 保存进度
                if resume and (success_count + fail_count) % 10 == 0:
                    save_progress(progress_file, list(completed_symbols))
    else:
        # 单进程处理
        for args in args_list:
            symbol, success, record_count, feature_count, error_msg = process_single_feature(args)
            
            if success:
                print(f"[{success_count + fail_count + 1}/{total}] {symbol}: "
                      f"{record_count} 条记录, {feature_count} 个特征已缓存")
                success_count += 1
                completed_symbols.add(symbol)
            else:
                print(f"[{success_count + fail_count + 1}/{total}] {symbol}: 失败 - {error_msg}")
                fail_count += 1
            
            # 保存进度
            if resume and (success_count + fail_count) % 10 == 0:
                save_progress(progress_file, list(completed_symbols))
    
    # 保存最终进度
    if resume:
        save_progress(progress_file, list(completed_symbols))
    
    print(f"\n" + "=" * 70)
    print(f"特征数据预热完成")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    
    # 显示缓存信息
    feature_cache = FeatureCache(cache_dir=feature_cache_dir)
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
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发进程数，默认1（单进程）。推荐设置为CPU核心数"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续传，从上次中断的地方继续"
    )
    
    parser.add_argument(
        "--clear-progress",
        action="store_true",
        help="清除进度文件，重新开始"
    )
    
    args = parser.parse_args()
    
    # 清除进度
    if args.clear_progress:
        for progress_file in ["data/.progress_kline.json", "data/.progress_features.json"]:
            if Path(progress_file).exists():
                Path(progress_file).unlink()
                print(f"进度文件已清除: {progress_file}")
        return
    
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
            args.kline_cache_dir,
            workers=args.workers,
            resume=args.resume
        )
    
    # 预热特征数据
    if not args.kline_only:
        prepare_features(
            symbols,
            args.start_date,
            args.end_date,
            args.kline_cache_dir,
            args.feature_cache_dir,
            workers=args.workers,
            resume=args.resume
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
