"""
快速测试脚本 - 检查各模块是否正常导入和基本功能
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """测试模块导入"""
    print("=" * 60)
    print("  测试模块导入")
    print("=" * 60)
    
    modules_to_test = [
        ("数据加载", "src.data.loader", "StockDataLoader"),
        ("数据验证", "src.data.validator", "DataValidator"),
        ("数据预处理", "src.data.preprocessor", "DataPreprocessor"),
        ("特征工程", "src.features.engineer", "FeatureEngineer"),
        ("数据集构建", "src.features.dataset_builder", "FeatureDatasetBuilder"),
        ("LSTM模型", "src.models.lstm_model", "LSTMModel"),
        ("GRU模型", "src.models.gru_model", "GRUModel"),
        ("Transformer模型", "src.models.transformer_model", "TransformerModel"),
        ("模型训练器", "src.training.trainer", "ModelTrainer"),
        ("模型评估器", "src.training.evaluator", "ModelEvaluator"),
        ("预测引擎", "src.prediction.engine", "PredictionEngine"),
        ("技术指标计算", "src.indicators.calculator", "TechnicalIndicatorCalculator"),
        ("数据库模型", "src.database.models", "StockBasicInfo"),
        ("数据库连接", "src.database.connection", "get_db"),
        ("配置", "src.config", "settings"),
    ]
    
    success_count = 0
    fail_count = 0
    
    for name, module_path, class_name in modules_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✓ {name:20s} - {module_path}")
            success_count += 1
        except Exception as e:
            print(f"✗ {name:20s} - {module_path}: {e}")
            fail_count += 1
    
    print(f"\n总计: {success_count} 成功, {fail_count} 失败")
    return fail_count == 0


def test_database_connection():
    """测试数据库连接"""
    print("\n" + "=" * 60)
    print("  测试数据库连接")
    print("=" * 60)
    
    try:
        from src.database.connection import get_db_context
        from src.database.models import StockBasicInfo
        
        print("\n尝试连接数据库...")
        with get_db_context() as db:
            # 查询股票数量
            count = db.query(StockBasicInfo).count()
            print(f"✓ 数据库连接成功")
            print(f"✓ 股票基本信息表记录数: {count}")
            
            # 查询活跃股票数量
            active_count = db.query(StockBasicInfo).filter(
                StockBasicInfo.is_active == 1
            ).count()
            print(f"✓ 活跃股票数量: {active_count}")
            
            if active_count > 0:
                # 获取一只股票的信息
                stock = db.query(StockBasicInfo).filter(
                    StockBasicInfo.is_active == 1
                ).first()
                print(f"\n示例股票:")
                print(f"  代码: {stock.symbol}")
                print(f"  名称: {stock.name}")
                print(f"  行业: {stock.industry}")
                print(f"  市场: {stock.market}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据库连接失败: {e}")
        print("\n请检查:")
        print("  1. MySQL服务是否启动")
        print("  2. .env文件中的数据库配置是否正确")
        print("  3. 数据库是否已创建并导入数据")
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n" + "=" * 60)
    print("  测试模型创建")
    print("=" * 60)
    
    try:
        import torch
        from src.models.lstm_model import LSTMModel
        from src.models.gru_model import GRUModel
        from src.models.transformer_model import TransformerModel
        
        input_size = 50
        
        # LSTM
        print("\n创建LSTM模型...")
        lstm = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2)
        print(f"✓ LSTM模型参数数量: {sum(p.numel() for p in lstm.parameters()):,}")
        
        # GRU
        print("\n创建GRU模型...")
        gru = GRUModel(input_size=input_size, hidden_size=64, num_layers=2)
        print(f"✓ GRU模型参数数量: {sum(p.numel() for p in gru.parameters()):,}")
        
        # Transformer
        print("\n创建Transformer模型...")
        transformer = TransformerModel(input_size=input_size, d_model=64, nhead=4, num_layers=2)
        print(f"✓ Transformer模型参数数量: {sum(p.numel() for p in transformer.parameters()):,}")
        
        # 测试前向传播
        print("\n测试前向传播...")
        batch_size = 8
        seq_length = 60
        x = torch.randn(batch_size, seq_length, input_size)
        
        with torch.no_grad():
            lstm_out = lstm(x)
            gru_out = gru(x)
            transformer_out = transformer(x)
        
        print(f"✓ LSTM输出形状: {lstm_out.shape}")
        print(f"✓ GRU输出形状: {gru_out.shape}")
        print(f"✓ Transformer输出形状: {transformer_out.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_engineering():
    """测试特征工程"""
    print("\n" + "=" * 60)
    print("  测试特征工程")
    print("=" * 60)
    
    try:
        import pandas as pd
        import numpy as np
        from src.features.engineer import FeatureEngineer
        
        # 创建模拟数据
        print("\n创建模拟K线数据...")
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        df = pd.DataFrame({
            'open': 10 + np.random.randn(100).cumsum() * 0.1,
            'high': 10.5 + np.random.randn(100).cumsum() * 0.1,
            'low': 9.5 + np.random.randn(100).cumsum() * 0.1,
            'close': 10 + np.random.randn(100).cumsum() * 0.1,
            'vol': np.random.randint(1000000, 10000000, 100),
            'amount': np.random.randint(10000000, 100000000, 100)
        }, index=dates)
        
        print(f"✓ 模拟数据形状: {df.shape}")
        
        # 特征工程
        engineer = FeatureEngineer()
        
        print("\n计算技术指标...")
        df_indicators = engineer.create_technical_indicators(df)
        print(f"✓ 添加技术指标后: {df_indicators.shape}")
        
        print("\n计算价格特征...")
        df_price = engineer.create_price_features(df_indicators)
        print(f"✓ 添加价格特征后: {df_price.shape}")
        
        print("\n计算成交量特征...")
        df_volume = engineer.create_volume_features(df_price)
        print(f"✓ 添加成交量特征后: {df_volume.shape}")
        
        print(f"\n生成的特征数量: {df_volume.shape[1] - df.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"✗ 特征工程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  A股AI模型系统 - 快速测试")
    print("=" * 60)
    
    results = []
    
    # 测试1: 模块导入
    results.append(("模块导入", test_imports()))
    
    # 测试2: 数据库连接
    results.append(("数据库连接", test_database_connection()))
    
    # 测试3: 模型创建
    results.append(("模型创建", test_model_creation()))
    
    # 测试4: 特征工程
    results.append(("特征工程", test_feature_engineering()))
    
    # 总结
    print("\n" + "=" * 60)
    print("  测试总结")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    print(f"\n总计: {success_count}/{total_count} 测试通过")
    
    if success_count == total_count:
        print("\n✓ 所有测试通过！系统准备就绪。")
        print("\n下一步:")
        print("  运行完整测试: python test_training_prediction.py")
    else:
        print("\n⚠ 部分测试失败，请检查配置和依赖。")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
