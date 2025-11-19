"""调试数据问题"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import StockDataLoader
from src.features.engineer import FeatureEngineer
import pandas as pd

def main():
    print("=" * 70)
    print("  调试数据问题")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n1. 加载原始数据...")
    loader = StockDataLoader()
    df = loader.load_kline_data("000001", "2021-01-01", "2024-12-31")
    print(f"✓ 数据量: {len(df)} 条")
    print(f"✓ 日期范围: {df.index[0]} 至 {df.index[-1]}")
    print(f"✓ 列名: {list(df.columns)}")
    print(f"\n前5行数据:")
    print(df.head())
    
    # 检查NaN
    print(f"\n原始数据NaN统计:")
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print(nan_counts[nan_counts > 0])
    else:
        print("  无NaN")
    
    # 2. 添加技术指标
    print("\n2. 添加技术指标...")
    engineer = FeatureEngineer()
    df_tech = engineer.create_technical_indicators(df)
    print(f"✓ 添加技术指标后: {len(df_tech)} 条, {len(df_tech.columns)} 列")
    
    # 检查NaN
    print(f"\n技术指标后NaN统计:")
    nan_counts = df_tech.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0].sort_values(ascending=False)
    print(f"包含NaN的列数: {len(cols_with_nan)}")
    print(f"\n前10个NaN最多的列:")
    for col, count in cols_with_nan.head(10).items():
        print(f"  {col}: {count} ({count/len(df_tech)*100:.1f}%)")
    
    # 3. 添加价格特征
    print("\n3. 添加价格特征...")
    df_price = engineer.create_price_features(df_tech)
    print(f"✓ 添加价格特征后: {len(df_price)} 条, {len(df_price.columns)} 列")
    
    # 检查NaN
    print(f"\n价格特征后NaN统计:")
    nan_counts = df_price.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0].sort_values(ascending=False)
    print(f"包含NaN的列数: {len(cols_with_nan)}")
    print(f"\n前10个NaN最多的列:")
    for col, count in cols_with_nan.head(10).items():
        print(f"  {col}: {count} ({count/len(df_price)*100:.1f}%)")
    
    # 4. 添加成交量特征
    print("\n4. 添加成交量特征...")
    df_vol = engineer.create_volume_features(df_price)
    print(f"✓ 添加成交量特征后: {len(df_vol)} 条, {len(df_vol.columns)} 列")
    
    # 检查NaN
    print(f"\n成交量特征后NaN统计:")
    nan_counts = df_vol.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0].sort_values(ascending=False)
    print(f"包含NaN的列数: {len(cols_with_nan)}")
    print(f"\n前10个NaN最多的列:")
    for col, count in cols_with_nan.head(10).items():
        print(f"  {col}: {count} ({count/len(df_vol)*100:.1f}%)")
    
    # 5. 删除NaN
    print("\n5. 删除NaN行...")
    df_clean = df_vol.dropna()
    print(f"✓ 删除NaN后: {len(df_clean)} 条")
    print(f"✓ 删除了: {len(df_vol) - len(df_clean)} 条 ({(len(df_vol) - len(df_clean))/len(df_vol)*100:.1f}%)")
    
    if len(df_clean) > 0:
        print(f"\n清理后的数据:")
        print(f"  日期范围: {df_clean.index[0]} 至 {df_clean.index[-1]}")
        print(f"  列数: {len(df_clean.columns)}")
    else:
        print("\n⚠️  所有数据都被删除了！")
        print("\n分析原因:")
        # 检查每一行有多少NaN
        nan_per_row = df_vol.isna().sum(axis=1)
        print(f"  每行NaN数量统计:")
        print(f"    最小: {nan_per_row.min()}")
        print(f"    最大: {nan_per_row.max()}")
        print(f"    平均: {nan_per_row.mean():.2f}")
        print(f"    总列数: {len(df_vol.columns)}")
        
        # 找出没有任何完整行的原因
        print(f"\n  检查是否有完整的行（无NaN）:")
        complete_rows = df_vol.dropna()
        print(f"    完整行数: {len(complete_rows)}")
        
        if len(complete_rows) == 0:
            print(f"\n  所有行都至少有1个NaN！")
            print(f"  检查哪些列导致所有行都有NaN:")
            # 找出在所有行都有NaN的列
            all_nan_cols = nan_counts[nan_counts == len(df_vol)]
            if len(all_nan_cols) > 0:
                print(f"\n  完全是NaN的列 ({len(all_nan_cols)}个):")
                for col in all_nan_cols.index:
                    print(f"    - {col}")

if __name__ == "__main__":
    main()
