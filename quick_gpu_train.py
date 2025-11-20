"""快速GPU训练脚本 - 使用缓存数据"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.cached_loader import FeatureCache
from src.models.lstm_model import LSTMModel
from src.training.trainer import ModelTrainer
from src.training.evaluator import ModelEvaluator
from src.data.preprocessor import DataPreprocessor
import time

def main():
    print("=" * 70)
    print("🚀 快速GPU训练 - 使用缓存数据")
    print("=" * 70)
    
    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️  未检测到GPU，将使用CPU训练")
    
    # 配置
    symbol = "000001"  # 可以修改为其他股票代码
    seq_length = 60
    batch_size = 64
    epochs = 50
    
    print(f"\n训练配置:")
    print(f"  股票代码: {symbol}")
    print(f"  序列长度: {seq_length}")
    print(f"  批次大小: {batch_size}")
    print(f"  训练轮数: {epochs}")
    
    # 1. 从缓存加载特征数据
    print(f"\n{'='*70}")
    print("📂 从缓存加载特征数据...")
    print(f"{'='*70}")
    
    cache = FeatureCache(cache_dir="data/features")
    df_features = cache.load(symbol)
    
    if df_features is None or len(df_features) == 0:
        print(f"❌ 未找到 {symbol} 的缓存数据")
        print(f"\n可用的缓存股票:")
        cache_files = list(Path("data/features").glob("*_features.parquet"))
        for i, f in enumerate(cache_files[:10], 1):
            stock_code = f.stem.replace("_features", "")
            print(f"  {i}. {stock_code}")
        if len(cache_files) > 10:
            print(f"  ... 还有 {len(cache_files) - 10} 只股票")
        return
    
    print(f"✓ 加载成功: {len(df_features)} 条记录")
    print(f"✓ 特征数量: {len(df_features.columns)} 个")
    print(f"✓ 数据范围: {df_features.index[0]} 至 {df_features.index[-1]}")
    
    # 2. 准备训练数据
    print(f"\n{'='*70}")
    print("🔧 准备训练数据...")
    print(f"{'='*70}")
    
    # 排除非特征列
    exclude_cols = ['symbol']
    if 'symbol' in df_features.columns:
        df_features = df_features.drop(columns=['symbol'])
    
    # 确保有close列
    if 'close' not in df_features.columns:
        print("❌ 数据中没有close列")
        return
    
    # 划分数据集
    n_samples = len(df_features)
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)
    
    train_df = df_features.iloc[:train_end].copy()
    val_df = df_features.iloc[train_end:val_end].copy()
    test_df = df_features.iloc[val_end:].copy()
    
    print(f"✓ 训练集: {len(train_df)} 条")
    print(f"✓ 验证集: {len(val_df)} 条")
    print(f"✓ 测试集: {len(test_df)} 条")
    
    # 3. 创建序列数据
    preprocessor = DataPreprocessor()
    
    # 标准化训练集
    feature_cols = [col for col in train_df.columns if col != 'close']
    train_df_norm = preprocessor.normalize_features(train_df, method='standard', fit=True)
    
    # 创建训练序列
    X_train, y_train = preprocessor.create_sequences(
        train_df_norm,
        seq_length=seq_length,
        target_col='close',
        feature_cols=feature_cols
    )
    
    # 标准化验证集（使用训练集的scaler）
    val_df_norm = preprocessor.normalize_features(val_df, method='standard', fit=False)
    X_val, y_val = preprocessor.create_sequences(
        val_df_norm,
        seq_length=seq_length,
        target_col='close',
        feature_cols=feature_cols
    )
    
    # 标准化测试集
    test_df_norm = preprocessor.normalize_features(test_df, method='standard', fit=False)
    X_test, y_test = preprocessor.create_sequences(
        test_df_norm,
        seq_length=seq_length,
        target_col='close',
        feature_cols=feature_cols
    )
    
    print(f"\n✓ 序列数据准备完成:")
    print(f"  训练: X={X_train.shape}, y={y_train.shape}")
    print(f"  验证: X={X_val.shape}, y={y_val.shape}")
    print(f"  测试: X={X_test.shape}, y={y_test.shape}")
    
    # 4. 创建模型
    print(f"\n{'='*70}")
    print("🤖 创建LSTM模型...")
    print(f"{'='*70}")
    
    input_size = X_train.shape[2]
    model = LSTMModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    
    print(f"✓ 模型创建成功")
    print(f"  输入维度: {input_size}")
    print(f"  隐藏层大小: 128")
    print(f"  层数: 2")
    
    # 5. 训练模型
    print(f"\n{'='*70}")
    print("🏋️  开始训练...")
    print(f"{'='*70}")
    
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        batch_size=batch_size
    )
    
    start_time = time.time()
    
    history = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    print(f"\n✓ 训练完成!")
    print(f"  总耗时: {training_time:.2f} 秒")
    print(f"  平均每轮: {training_time/epochs:.2f} 秒")
    
    # 6. 评估模型
    print(f"\n{'='*70}")
    print("📊 评估模型...")
    print(f"{'='*70}")
    
    evaluator = ModelEvaluator(model=model, device=device)
    
    # 测试集评估
    test_metrics = evaluator.evaluate(X_test, y_test)
    
    print(f"\n测试集结果:")
    print(f"  MSE:  {test_metrics['mse']:.6f}")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  MAE:  {test_metrics['mae']:.6f}")
    print(f"  R²:   {test_metrics['r2']:.6f}")
    
    # 7. 保存模型
    output_dir = Path(f"out/{symbol}_lstm_quick")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': 128,
        'num_layers': 2,
        'seq_length': seq_length,
        'feature_cols': feature_cols,
        'scaler': preprocessor.scaler,
        'metrics': test_metrics,
        'history': history
    }, model_path)
    
    print(f"\n✓ 模型已保存: {model_path}")
    
    print(f"\n{'='*70}")
    print("✅ 训练完成!")
    print(f"{'='*70}")
    
    # 显示训练历史
    print(f"\n训练历史:")
    print(f"  最佳验证损失: {min(history['val_loss']):.6f} (第 {history['val_loss'].index(min(history['val_loss']))+1} 轮)")
    print(f"  最终训练损失: {history['train_loss'][-1]:.6f}")
    print(f"  最终验证损失: {history['val_loss'][-1]:.6f}")

if __name__ == "__main__":
    main()
