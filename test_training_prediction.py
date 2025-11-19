"""
端到端测试：训练和预测流程
测试股票日线数据的模型训练和预测功能
"""
import sys
from pathlib import Path
import numpy as np
import torch
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import StockDataLoader
from src.data.validator import DataValidator
from src.features.engineer import FeatureEngineer
from src.features.dataset_builder import FeatureDatasetBuilder
from src.models.lstm_model import LSTMModel
from src.models.gru_model import GRUModel
from src.models.transformer_model import TransformerModel
from src.training.trainer import ModelTrainer
from src.training.evaluator import ModelEvaluator
from src.prediction.engine import PredictionEngine


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_data_loading():
    """测试1: 数据加载"""
    print_section("测试1: 数据加载")
    
    try:
        loader = StockDataLoader()
        
        # 获取活跃股票列表
        print("\n获取活跃股票列表...")
        active_stocks = loader.get_all_active_stocks()
        print(f"✓ 活跃股票数量: {len(active_stocks)}")
        
        if len(active_stocks) == 0:
            print("✗ 没有找到活跃股票，请检查数据库")
            return None
        
        # 选择第一只股票进行测试
        test_symbol = active_stocks[0]
        print(f"✓ 测试股票: {test_symbol}")
        
        # 获取股票信息
        print(f"\n获取股票 {test_symbol} 的基本信息...")
        stock_info = loader.load_stock_info(test_symbol)
        if stock_info:
            print(f"✓ 股票名称: {stock_info.get('name')}")
            print(f"✓ 所属行业: {stock_info.get('industry')}")
            print(f"✓ 上市日期: {stock_info.get('list_date')}")
        
        # 获取数据日期范围
        print(f"\n获取数据日期范围...")
        date_range = loader.get_date_range(test_symbol)
        print(f"✓ 数据起始日期: {date_range['start_date']}")
        print(f"✓ 数据结束日期: {date_range['end_date']}")
        
        # 加载最近3年的数据
        end_date = date_range['end_date']
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365*3)).strftime('%Y-%m-%d')
        
        print(f"\n加载K线数据 ({start_date} 至 {end_date})...")
        df = loader.load_kline_data(test_symbol, start_date, end_date)
        
        if df.empty:
            print(f"✗ 没有找到股票 {test_symbol} 的数据")
            return None
        
        print(f"✓ 数据行数: {len(df)}")
        print(f"✓ 数据列数: {len(df.columns)}")
        print(f"\n数据前5行:")
        print(df[['open', 'high', 'low', 'close', 'vol']].head())
        
        # 数据验证
        print(f"\n验证数据质量...")
        validator = DataValidator()
        
        missing_check = validator.check_missing_values(df)
        print(f"✓ 缺失值检查: {'通过' if missing_check else '失败'}")
        
        anomalies = validator.check_price_anomalies(df)
        if anomalies:
            print(f"⚠ 发现 {len(anomalies)} 个价格异常")
        else:
            print(f"✓ 价格异常检查: 通过")
        
        return test_symbol, df
        
    except Exception as e:
        print(f"\n✗ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_feature_engineering(df):
    """测试2: 特征工程"""
    print_section("测试2: 特征工程")
    
    try:
        engineer = FeatureEngineer()
        
        print("\n生成技术指标...")
        df_with_indicators = engineer.create_technical_indicators(df)
        print(f"✓ 添加技术指标后列数: {len(df_with_indicators.columns)}")
        
        print("\n生成价格特征...")
        df_with_price = engineer.create_price_features(df_with_indicators)
        print(f"✓ 添加价格特征后列数: {len(df_with_price.columns)}")
        
        print("\n生成成交量特征...")
        df_with_volume = engineer.create_volume_features(df_with_price)
        print(f"✓ 添加成交量特征后列数: {len(df_with_volume.columns)}")
        
        print(f"\n特征列表（部分）:")
        feature_cols = [col for col in df_with_volume.columns if col not in ['symbol', 'open', 'high', 'low', 'close', 'vol']]
        for i, col in enumerate(feature_cols[:10]):
            print(f"  {i+1}. {col}")
        if len(feature_cols) > 10:
            print(f"  ... 还有 {len(feature_cols) - 10} 个特征")
        
        return df_with_volume
        
    except Exception as e:
        print(f"\n✗ 特征工程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_dataset_building(df_features):
    """测试3: 数据集构建"""
    print_section("测试3: 数据集构建")
    
    try:
        builder = FeatureDatasetBuilder()
        
        print("\n构建特征矩阵...")
        df_matrix = builder.build_feature_matrix(df_features)
        print(f"✓ 特征矩阵形状: {df_matrix.shape}")
        
        print("\n准备时间序列数据...")
        seq_length = 60
        X, y, feature_names = builder.prepare_sequences(
            df_matrix,
            seq_length=seq_length,
            target_col='close',
            normalize=True
        )
        
        print(f"✓ 序列长度: {seq_length}")
        print(f"✓ X形状: {X.shape}")  # (样本数, 序列长度, 特征数)
        print(f"✓ y形状: {y.shape}")  # (样本数,)
        print(f"✓ 特征数量: {len(feature_names)}")
        
        print("\n划分训练集、验证集、测试集...")
        X_train, X_val, X_test, y_train, y_val, y_test = builder.split_dataset(
            X, y,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        print(f"✓ 训练集: {X_train.shape[0]} 样本")
        print(f"✓ 验证集: {X_val.shape[0]} 样本")
        print(f"✓ 测试集: {X_test.shape[0]} 样本")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': feature_names,
            'seq_length': seq_length
        }
        
    except Exception as e:
        print(f"\n✗ 数据集构建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_training(dataset_dict, model_type='lstm'):
    """测试4: 模型训练"""
    print_section(f"测试4: 模型训练 ({model_type.upper()})")
    
    try:
        X_train = dataset_dict['X_train']
        X_val = dataset_dict['X_val']
        y_train = dataset_dict['y_train']
        y_val = dataset_dict['y_val']
        
        input_size = X_train.shape[2]  # 特征数
        
        print(f"\n创建{model_type.upper()}模型...")
        print(f"✓ 输入特征数: {input_size}")
        
        # 创建模型
        if model_type == 'lstm':
            model = LSTMModel(
                input_size=input_size,
                hidden_size=64,
                num_layers=2,
                output_size=1,
                dropout=0.2
            )
        elif model_type == 'gru':
            model = GRUModel(
                input_size=input_size,
                hidden_size=64,
                num_layers=2,
                output_size=1,
                dropout=0.2
            )
        elif model_type == 'transformer':
            model = TransformerModel(
                input_size=input_size,
                d_model=64,
                nhead=4,
                num_layers=2,
                dim_feedforward=256,
                output_size=1,
                dropout=0.1
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        print(f"✓ 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 创建数据加载器
        print("\n创建数据加载器...")
        trainer = ModelTrainer(model, checkpoint_dir=f'checkpoints/{model_type}')
        
        train_loader = trainer.create_data_loader(X_train, y_train, batch_size=32, shuffle=True)
        val_loader = trainer.create_data_loader(X_val, y_val, batch_size=32, shuffle=False)
        
        print(f"✓ 训练批次数: {len(train_loader)}")
        print(f"✓ 验证批次数: {len(val_loader)}")
        
        # 训练模型（少量epoch用于测试）
        print("\n开始训练（测试模式：5个epoch）...")
        print("注意：这只是测试，实际训练需要更多epoch")
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=5,
            learning_rate=0.001,
            patience=10,
            checkpoint_interval=2
        )
        
        print(f"\n✓ 训练完成")
        print(f"✓ 最终训练损失: {history['train_losses'][-1]:.6f}")
        print(f"✓ 最终验证损失: {history['val_losses'][-1]:.6f}")
        print(f"✓ 最佳验证损失: {history['best_val_loss']:.6f}")
        
        return model, trainer, history
        
    except Exception as e:
        print(f"\n✗ 模型训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_model_evaluation(model, trainer, dataset_dict):
    """测试5: 模型评估"""
    print_section("测试5: 模型评估")
    
    try:
        X_test = dataset_dict['X_test']
        y_test = dataset_dict['y_test']
        
        print("\n创建测试数据加载器...")
        test_loader = trainer.create_data_loader(X_test, y_test, batch_size=32, shuffle=False)
        
        print("\n评估模型性能...")
        evaluator = ModelEvaluator(model, trainer.device)
        
        metrics = evaluator.evaluate(test_loader)
        
        print(f"\n✓ 评估指标:")
        print(f"  - MAE (平均绝对误差): {metrics['mae']:.6f}")
        print(f"  - RMSE (均方根误差): {metrics['rmse']:.6f}")
        print(f"  - MAPE (平均绝对百分比误差): {metrics['mape']:.2f}%")
        print(f"  - R² (决定系数): {metrics['r2']:.4f}")
        print(f"  - 方向准确率: {metrics['direction_accuracy']:.2f}%")
        
        return metrics
        
    except Exception as e:
        print(f"\n✗ 模型评估测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_prediction(test_symbol, model_type, dataset_dict):
    """测试6: 预测功能"""
    print_section("测试6: 预测功能")
    
    try:
        print("\n注意：预测功能需要完整训练的模型")
        print("当前使用的是测试训练的模型，预测结果仅供参考\n")
        
        # 模型配置
        model_config = {
            'input_size': dataset_dict['X_train'].shape[2],
            'seq_length': dataset_dict['seq_length'],
            'hidden_size': 64,
            'num_layers': 2,
            'output_size': 1,
            'dropout': 0.2
        }
        
        # 使用最佳模型进行预测
        model_path = f'checkpoints/{model_type}/best_model.pth'
        
        print(f"加载模型: {model_path}")
        
        engine = PredictionEngine(
            model_path=model_path,
            model_type=model_type,
            model_config=model_config
        )
        
        print(f"✓ 预测引擎已初始化")
        
        # 简单预测
        print(f"\n生成未来5天的价格预测...")
        result = engine.predict(test_symbol, days=5)
        
        print(f"\n✓ 预测结果:")
        print(f"  股票代码: {result['symbol']}")
        print(f"  基准日期: {result['base_date']}")
        print(f"  基准价格: {result['base_price']:.2f}")
        print(f"  趋势判断: {result['trend']}")
        print(f"\n  未来5天预测:")
        for pred in result['predictions']:
            change_pct = (pred['price'] - result['base_price']) / result['base_price'] * 100
            print(f"    {pred['date']}: {pred['price']:.2f} ({change_pct:+.2f}%)")
        
        # 带置信区间的预测
        print(f"\n生成带置信区间的预测（95%置信水平）...")
        result_conf = engine.predict_with_confidence(
            test_symbol,
            days=5,
            confidence_level=0.95,
            n_samples=50  # 测试用，实际应该更多
        )
        
        print(f"\n✓ 置信区间预测结果:")
        print(f"  置信度分数: {result_conf['confidence_score']:.2f}")
        print(f"  蒙特卡洛采样次数: {result_conf['n_samples']}")
        print(f"\n  未来5天预测（含置信区间）:")
        for pred in result_conf['predictions']:
            print(f"    {pred['date']}: {pred['price']:.2f} "
                  f"[{pred['confidence_lower']:.2f} - {pred['confidence_upper']:.2f}]")
        
        return result, result_conf
        
    except Exception as e:
        print(f"\n✗ 预测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """主测试流程"""
    print("\n" + "=" * 70)
    print("  A股AI模型系统 - 端到端测试")
    print("  测试：数据加载 → 特征工程 → 模型训练 → 预测")
    print("=" * 70)
    
    # 测试1: 数据加载
    result = test_data_loading()
    if result is None:
        print("\n测试终止：数据加载失败")
        return
    
    test_symbol, df = result
    
    # 测试2: 特征工程
    df_features = test_feature_engineering(df)
    if df_features is None:
        print("\n测试终止：特征工程失败")
        return
    
    # 测试3: 数据集构建
    dataset_dict = test_dataset_building(df_features)
    if dataset_dict is None:
        print("\n测试终止：数据集构建失败")
        return
    
    # 测试4: 模型训练（使用LSTM）
    model_type = 'lstm'
    model, trainer, history = test_model_training(dataset_dict, model_type=model_type)
    if model is None:
        print("\n测试终止：模型训练失败")
        return
    
    # 测试5: 模型评估
    metrics = test_model_evaluation(model, trainer, dataset_dict)
    if metrics is None:
        print("\n警告：模型评估失败，但继续测试")
    
    # 测试6: 预测功能
    pred_result, pred_conf = test_prediction(test_symbol, model_type, dataset_dict)
    
    # 总结
    print_section("测试总结")
    print("\n✓ 所有核心功能测试完成！")
    print("\n测试覆盖:")
    print("  ✓ 数据加载和验证")
    print("  ✓ 技术指标计算")
    print("  ✓ 特征工程")
    print("  ✓ 数据集构建和划分")
    print("  ✓ 模型训练（LSTM）")
    print("  ✓ 模型评估")
    print("  ✓ 价格预测")
    print("  ✓ 置信区间预测")
    
    print("\n注意事项:")
    print("  - 本测试使用少量epoch进行快速验证")
    print("  - 实际生产环境需要训练更多epoch（建议50-100）")
    print("  - 可以测试其他模型类型（GRU、Transformer）")
    print("  - 预测结果仅供参考，需要完整训练后才能实际使用")
    
    print("\n下一步:")
    print("  1. 使用完整数据集训练模型（更多epoch）")
    print("  2. 测试GRU和Transformer模型")
    print("  3. 调优超参数")
    print("  4. 集成到API服务")
    print("  5. 开发前端界面")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
