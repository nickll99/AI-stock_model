"""
批量训练全市场股票模型

使用方法:
    # 训练所有股票
    python scripts/batch_train_all_stocks.py
    
    # 训练指定股票
    python scripts/batch_train_all_stocks.py --symbols 000001,600519,000858
    
    # 使用多进程并发训练
    python scripts/batch_train_all_stocks.py --workers 4
    
    # 断点续传
    python scripts/batch_train_all_stocks.py --resume
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
import torch

from src.data.loader import StockDataLoader
from src.features.dataset_builder import FeatureDatasetBuilder
from src.models.lstm_model import LSTMModel
from src.models.gru_model import GRUModel
from src.models.transformer_model import TransformerModel
from src.training.trainer import ModelTrainer
from src.training.evaluator import ModelEvaluator
from src.training.training_manager import TrainingManager


def train_single_stock(args: Tuple) -> Tuple[str, bool, dict, str]:
    """
    训练单只股票（用于多进程）
    
    Args:
        args: (symbol, config, index, total)
        
    Returns:
        (symbol, success, metrics, message)
    """
    symbol, config, index, total = args
    
    try:
        print(f"\n{'='*70}")
        print(f"[{index}/{total}] 开始训练: {symbol}")
        print(f"{'='*70}")
        
        # 1. 初始化训练管理器
        manager = TrainingManager(
            stock_symbol=symbol,
            model_type=config['model_type'],
            output_dir=config['output_dir']
        )
        
        # 2. 加载数据
        loader = StockDataLoader()
        df = loader.load_kline_data(
            symbol,
            config['train_start_date'],
            config['train_end_date']
        )
        
        if len(df) < 200:
            return (symbol, False, {}, f"数据量不足: {len(df)} 条")
        
        print(f"✓ 加载数据: {len(df)} 条记录")
        
        # 3. 构建特征和数据集
        builder = FeatureDatasetBuilder()
        dataset = builder.build_complete_dataset(
            df,
            seq_length=config['seq_length'],
            target_col='close',
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            normalize=True,
            use_cache=False
        )
        
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_val = dataset['X_val']
        y_val = dataset['y_val']
        X_test = dataset['X_test']
        y_test = dataset['y_test']
        
        print(f"✓ 训练集: {X_train.shape[0]} 样本")
        print(f"✓ 验证集: {X_val.shape[0]} 样本")
        print(f"✓ 测试集: {X_test.shape[0]} 样本")
        
        # 4. 创建模型
        input_size = X_train.shape[2]
        
        if config['model_type'] == 'lstm':
            model = LSTMModel(
                input_size=input_size,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            )
        elif config['model_type'] == 'gru':
            model = GRUModel(
                input_size=input_size,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            )
        else:
            model = TransformerModel(
                input_size=input_size,
                d_model=config['hidden_size'],
                nhead=4,
                num_layers=config['num_layers'],
                dropout=config['dropout']
            )
        
        # 5. 训练模型
        trainer = ModelTrainer(model, manager.run_dir)
        
        history = trainer.train(
            X_train, y_train,
            X_val, y_val,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            patience=config['patience']
        )
        
        print(f"✓ 训练完成")
        print(f"✓ 最佳验证损失: {history['best_val_loss']:.6f}")
        
        # 6. 评估模型
        evaluator = ModelEvaluator(model, trainer.device)
        metrics = evaluator.evaluate(X_test, y_test, batch_size=config['batch_size'])
        
        print(f"✓ 评估指标:")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  R²: {metrics['r2']:.6f}")
        
        # 7. 保存结果
        manager.save_training_history(history)
        manager.save_evaluation_metrics(metrics)
        manager.save_model_info(model, input_size)
        
        return (symbol, True, metrics, "成功")
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"✗ 训练失败: {symbol} - {e}")
        return (symbol, False, {}, error_msg)


def save_progress(progress_file: str, completed: List[str], failed: List[str]):
    """保存训练进度"""
    Path(progress_file).parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump({
            'completed': completed,
            'failed': failed,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)


def load_progress(progress_file: str) -> Tuple[set, set]:
    """加载训练进度"""
    if not Path(progress_file).exists():
        return set(), set()
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)
            return set(progress.get('completed', [])), set(progress.get('failed', []))
    except Exception as e:
        print(f"加载进度文件失败: {e}")
        return set(), set()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量训练全市场股票模型")
    
    parser.add_argument(
        "--symbols",
        type=str,
        default="all",
        help="股票代码，用逗号分隔，或使用'all'表示所有活跃股票"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="lstm",
        choices=["lstm", "gru", "transformer"],
        help="模型类型"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发进程数，默认1（单进程）"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续传，跳过已完成的股票"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out",
        help="输出目录"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="训练轮数"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批次大小"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制训练的股票数量（用于测试）"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  批量训练全市场股票模型")
    print("=" * 70)
    
    # 配置
    config = {
        "model_type": args.model_type,
        "seq_length": 60,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": 0.001,
        "patience": 10,
        "train_start_date": "2021-01-01",
        "train_end_date": "2024-12-31",
        "output_dir": args.output_dir
    }
    
    print(f"\n训练配置:")
    print(f"  模型类型: {config['model_type']}")
    print(f"  序列长度: {config['seq_length']}")
    print(f"  训练轮数: {config['epochs']}")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  并发进程: {args.workers}")
    print(f"  断点续传: {'是' if args.resume else '否'}")
    
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
        print(f"限制训练数量: {args.limit}")
    
    # 加载进度
    progress_file = f"{args.output_dir}/.batch_training_progress.json"
    completed_symbols, failed_symbols = set(), set()
    
    if args.resume:
        completed_symbols, failed_symbols = load_progress(progress_file)
        print(f"\n断点续传:")
        print(f"  已完成: {len(completed_symbols)} 只")
        print(f"  已失败: {len(failed_symbols)} 只")
    
    # 过滤已完成的股票
    remaining_symbols = [s for s in symbols if s not in completed_symbols]
    
    if not remaining_symbols:
        print("\n所有股票已完成训练！")
        return
    
    print(f"\n待训练股票: {len(remaining_symbols)} 只")
    
    # 准备参数
    total = len(symbols)
    args_list = [
        (symbol, config, i + len(completed_symbols) + 1, total)
        for i, symbol in enumerate(remaining_symbols)
    ]
    
    # 统计
    success_count = len(completed_symbols)
    fail_count = len(failed_symbols)
    start_time = time.time()
    
    # 训练
    print(f"\n{'='*70}")
    print(f"  开始批量训练")
    print(f"{'='*70}\n")
    
    if args.workers > 1:
        # 多进程训练
        print(f"使用 {args.workers} 个进程并行训练...\n")
        
        with Pool(args.workers) as pool:
            for result in pool.imap_unordered(train_single_stock, args_list):
                symbol, success, metrics, message = result
                
                if success:
                    success_count += 1
                    completed_symbols.add(symbol)
                    print(f"\n[{success_count + fail_count}/{total}] ✓ {symbol} 训练成功")
                    print(f"  MAE: {metrics['mae']:.6f}, RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.6f}")
                else:
                    fail_count += 1
                    failed_symbols.add(symbol)
                    print(f"\n[{success_count + fail_count}/{total}] ✗ {symbol} 训练失败: {message}")
                
                # 保存进度
                if args.resume and (success_count + fail_count) % 10 == 0:
                    save_progress(progress_file, list(completed_symbols), list(failed_symbols))
    else:
        # 单进程训练
        print("使用单进程顺序训练...\n")
        
        for args_item in args_list:
            symbol, success, metrics, message = train_single_stock(args_item)
            
            if success:
                success_count += 1
                completed_symbols.add(symbol)
                print(f"\n[{success_count + fail_count}/{total}] ✓ {symbol} 训练成功")
                print(f"  MAE: {metrics['mae']:.6f}, RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.6f}")
            else:
                fail_count += 1
                failed_symbols.add(symbol)
                print(f"\n[{success_count + fail_count}/{total}] ✗ {symbol} 训练失败: {message}")
            
            # 保存进度
            if args.resume and (success_count + fail_count) % 10 == 0:
                save_progress(progress_file, list(completed_symbols), list(failed_symbols))
    
    # 保存最终进度
    if args.resume:
        save_progress(progress_file, list(completed_symbols), list(failed_symbols))
    
    # 统计
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"  批量训练完成")
    print(f"{'='*70}")
    print(f"\n总结:")
    print(f"  总股票数: {total}")
    print(f"  成功: {success_count} ({success_count/total*100:.1f}%)")
    print(f"  失败: {fail_count} ({fail_count/total*100:.1f}%)")
    print(f"  总耗时: {elapsed_time/60:.1f} 分钟")
    print(f"  平均每只: {elapsed_time/total:.1f} 秒")
    
    if failed_symbols:
        print(f"\n失败的股票 ({len(failed_symbols)}只):")
        for symbol in list(failed_symbols)[:10]:
            print(f"  - {symbol}")
        if len(failed_symbols) > 10:
            print(f"  ... 还有 {len(failed_symbols) - 10} 只")
    
    print(f"\n输出目录: {args.output_dir}/")
    print()


if __name__ == "__main__":
    main()
