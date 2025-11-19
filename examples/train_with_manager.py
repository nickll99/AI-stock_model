"""
使用TrainingManager进行标准化训练的示例
数据来源：MySQL数据库
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datetime import datetime, timedelta

from src.data.loader import StockDataLoader
from src.features.dataset_builder import FeatureDatasetBuilder
from src.models.lstm_model import LSTMModel
from src.training.trainer import ModelTrainer
from src.training.evaluator import ModelEvaluator
from src.training.training_manager import TrainingManager


def main():
    """主训练流程"""
    print("=" * 70)
    print("  标准化训练流程示例")
    print("  数据来源: MySQL")
    print("  输出目录: out/")
    print("=" * 70)
    
    # ========== 1. 配置参数 ==========
    stock_symbol = "000001"  # 股票代码
    model_type = "lstm"      # 模型类型
    
    config = {
        "stock_symbol": stock_symbol,
        "model_type": model_type,
        "seq_length": 60,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "patience": 10,
        "data_source": "MySQL",
        "train_start_date": "2021-01-01",
        "train_end_date": "2024-12-31"
    }
    
    print(f"\n训练配置:")
    print(f"  股票代码: {stock_symbol}")
    print(f"  模型类型: {model_type}")
    print(f"  序列长度: {config['seq_length']}")
    print(f"  训练轮数: {config['epochs']}")
    
    # ========== 2. 初始化训练管理器 ==========
    print(f"\n初始化训练管理器...")
    manager = TrainingManager(
        stock_symbol=stock_symbol,
        model_type=model_type,
        output_dir="out"
    )
    
    print(f"✓ 训练运行: {manager.run_name}")
    print(f"✓ 输出目录: {manager.run_dir}")
    
    # 保存配置
    manager.save_config(config)
    manager.update_status("preparing")
    
    # ========== 3. 从MySQL加载数据 ==========
    print(f"\n从MySQL加载数据...")
    manager.save_training_log("开始加载数据")
    
    loader = StockDataLoader()
    
    # 获取数据日期范围
    date_range = loader.get_date_range(stock_symbol)
    print(f"✓ 数据范围: {date_range['start_date']} 至 {date_range['end_date']}")
    
    # 加载训练数据
    end_date = config['train_end_date']
    start_date = config['train_start_date']
    
    df = loader.load_kline_data(stock_symbol, start_date, end_date)
    print(f"✓ 加载数据: {len(df)} 条记录")
    
    manager.save_training_log(f"数据加载完成: {len(df)} 条记录")
    
    # ========== 4. 特征工程 ==========
    print(f"\n构建特征...")
    manager.save_training_log("开始特征工程")
    
    builder = FeatureDatasetBuilder()
    
    # 方法1：使用build_complete_dataset一步到位（推荐）
    # 这个方法会自动完成特征构建、数据划分、序列准备等所有步骤
    dataset = builder.build_complete_dataset(
        df,
        seq_length=config['seq_length'],
        target_col='close',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        normalize=True,
        use_cache=False  # 不使用缓存，确保使用最新数据
    )
    
    # 提取训练数据
    X = dataset['X_train']
    y = dataset['y_train']
    feature_names = dataset['feature_names']
    
    print(f"✓ 训练集: {X.shape}")
    print(f"✓ 验证集: {dataset['X_val'].shape}")
    print(f"✓ 测试集: {dataset['X_test'].shape}")
    print(f"✓ 特征数量: {len(feature_names)}")
    
    manager.save_training_log(f"特征工程完成: {X.shape}")
    
    # 数据集已经在build_complete_dataset中划分好了
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_val = dataset['X_val']
    y_val = dataset['y_val']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    
    print(f"✓ 训练集: {X_train.shape[0]} 样本")
    print(f"✓ 验证集: {X_val.shape[0]} 样本")
    print(f"✓ 测试集: {X_test.shape[0]} 样本")
    
    # ========== 5. 创建模型 ==========
    print(f"\n创建模型...")
    manager.save_training_log("创建模型")
    
    input_size = X_train.shape[2]
    
    model = LSTMModel(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=1,
        dropout=config['dropout']
    )
    
    print(f"✓ 模型类型: {model_type.upper()}")
    print(f"✓ 参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 保存模型配置
    model_config = {
        "model_type": model_type,
        "input_size": input_size,
        "hidden_size": config['hidden_size'],
        "num_layers": config['num_layers'],
        "output_size": 1,
        "dropout": config['dropout'],
        "feature_names": feature_names
    }
    
    # ========== 6. 训练模型 ==========
    print(f"\n开始训练...")
    manager.update_status("training", start_time=datetime.now().isoformat())
    manager.save_training_log("开始训练")
    
    trainer = ModelTrainer(
        model,
        checkpoint_dir=str(manager.run_dir / "checkpoints")
    )
    
    train_loader = trainer.create_data_loader(X_train, y_train, batch_size=config['batch_size'])
    val_loader = trainer.create_data_loader(X_val, y_val, batch_size=config['batch_size'])
    
    # 训练回调函数
    def training_callback(epoch, train_loss, val_loss):
        # 记录指标
        manager.log_metrics(epoch, {
            "train_loss": train_loss,
            "val_loss": val_loss
        }, phase="train")
        
        # 保存训练日志
        manager.save_training_log(
            f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
        )
    
    # 开始训练
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        checkpoint_interval=10,
        callback=training_callback
    )
    
    print(f"\n✓ 训练完成")
    print(f"✓ 最佳验证损失: {history['best_val_loss']:.6f}")
    
    manager.save_training_log(f"训练完成: best_val_loss={history['best_val_loss']:.6f}")
    
    # ========== 7. 保存模型 ==========
    print(f"\n保存模型...")
    
    training_args = {
        "epochs": config['epochs'],
        "batch_size": config['batch_size'],
        "learning_rate": config['learning_rate'],
        "optimizer": "Adam",
        "loss_function": "MSELoss",
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "best_val_loss": history['best_val_loss'],
        "total_epochs_trained": history['total_epochs']
    }
    
    manager.save_model(model, model_config, training_args)
    
    # ========== 8. 评估模型 ==========
    print(f"\n评估模型...")
    manager.save_training_log("开始评估")
    
    evaluator = ModelEvaluator(model, trainer.device)
    metrics = evaluator.evaluate(X_test, y_test, batch_size=config['batch_size'])
    
    print(f"\n✓ 评估指标:")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  方向准确率: {metrics['direction_accuracy']:.2f}%")
    
    manager.save_evaluation_results(metrics)
    manager.save_training_log(f"评估完成: MAE={metrics['mae']:.6f}")
    
    # ========== 9. 完成训练 ==========
    manager.update_status(
        "completed",
        end_time=datetime.now().isoformat(),
        final_metrics=metrics
    )
    
    # ========== 10. 输出摘要 ==========
    print(f"\n" + "=" * 70)
    print("  训练完成")
    print("=" * 70)
    
    summary = manager.get_summary()
    
    print(f"\n训练摘要:")
    print(f"  运行名称: {summary['run_name']}")
    print(f"  输出目录: {summary['run_dir']}")
    print(f"  模型大小: {summary.get('model_size_mb', 0):.2f} MB")
    print(f"  检查点数: {summary.get('num_checkpoints', 0)}")
    print(f"  状态: {summary['metadata']['status']}")
    
    print(f"\n输出文件:")
    print(f"  ✓ 模型文件: {manager.get_model_path()}")
    print(f"  ✓ 配置文件: {manager.run_dir / 'config.json'}")
    print(f"  ✓ 评估结果: {manager.run_dir / 'results' / 'eval_results.json'}")
    print(f"  ✓ 训练日志: {manager.run_dir / 'logs' / 'training.log'}")
    
    print(f"\n" + "=" * 70)


if __name__ == "__main__":
    main()
