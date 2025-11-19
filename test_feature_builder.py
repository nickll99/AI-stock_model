"""测试特征构建器"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import StockDataLoader
from src.features.dataset_builder import FeatureDatasetBuilder

def main():
    print("=" * 70)
    print("  测试特征构建器")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    loader = StockDataLoader()
    df = loader.load_kline_data("000001", "2021-01-01", "2024-12-31")
    print(f"✓ 加载数据: {len(df)} 条记录")
    print(f"✓ 日期范围: {df.index[0]} 至 {df.index[-1]}")
    
    # 2. 构建特征
    print("\n2. 构建特征...")
    builder = FeatureDatasetBuilder()
    
    try:
        df_features = builder.build_feature_matrix(df)
        print(f"✓ 特征矩阵: {df_features.shape}")
        print(f"✓ 特征列: {list(df_features.columns[:10])}... (共{len(df_features.columns)}列)")
    except Exception as e:
        print(f"✗ 特征构建失败: {e}")
        return
    
    # 3. 准备序列
    print("\n3. 准备序列...")
    try:
        X, y, feature_names = builder.prepare_sequences(
            df_features,
            seq_length=60,
            target_col='close',
            normalize=True
        )
        print(f"✓ X shape: {X.shape}")
        print(f"✓ y shape: {y.shape}")
        print(f"✓ 特征数量: {len(feature_names)}")
    except Exception as e:
        print(f"✗ 序列准备失败: {e}")
        return
    
    # 4. 使用build_complete_dataset（推荐方式）
    print("\n4. 使用build_complete_dataset...")
    try:
        dataset = builder.build_complete_dataset(
            df,
            seq_length=60,
            target_col='close',
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            normalize=True,
            use_cache=False
        )
        
        print(f"✓ 训练集: {dataset['X_train'].shape}")
        print(f"✓ 验证集: {dataset['X_val'].shape}")
        print(f"✓ 测试集: {dataset['X_test'].shape}")
        print(f"✓ 特征数量: {len(dataset['feature_names'])}")
        
    except Exception as e:
        print(f"✗ 完整数据集构建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("  测试完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
