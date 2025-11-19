"""
查看训练运行列表和详情
"""
import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.training_manager import TrainingManager


def format_datetime(dt_str):
    """格式化日期时间"""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return dt_str


def list_runs():
    """列出所有训练运行"""
    print("=" * 80)
    print("  训练运行列表")
    print("=" * 80)
    
    runs = TrainingManager.list_runs()
    
    if not runs:
        print("\n没有找到训练运行")
        print("提示: 运行 python examples/train_with_manager.py 开始训练")
        return
    
    print(f"\n找到 {len(runs)} 个训练运行:\n")
    
    for i, run in enumerate(runs, 1):
        print(f"{i}. {run['run_name']}")
        print(f"   股票代码: {run.get('stock_symbol', 'N/A')}")
        print(f"   模型类型: {run.get('model_type', 'N/A')}")
        print(f"   状态: {run.get('status', 'unknown')}")
        print(f"   创建时间: {format_datetime(run.get('created_at', ''))}")
        
        if 'eval_metrics' in run:
            metrics = run['eval_metrics']
            print(f"   评估指标:")
            print(f"     - MAE: {metrics.get('mae', 0):.6f}")
            print(f"     - RMSE: {metrics.get('rmse', 0):.6f}")
            print(f"     - MAPE: {metrics.get('mape', 0):.2f}%")
            print(f"     - R²: {metrics.get('r2', 0):.4f}")
        
        print()


def show_run_details(run_name: str):
    """显示训练运行详情"""
    print("=" * 80)
    print(f"  训练运行详情: {run_name}")
    print("=" * 80)
    
    try:
        manager = TrainingManager.load_run(run_name)
        summary = manager.get_summary()
        
        print(f"\n基本信息:")
        print(f"  运行名称: {summary['run_name']}")
        print(f"  输出目录: {summary['run_dir']}")
        print(f"  股票代码: {summary['metadata'].get('stock_symbol', 'N/A')}")
        print(f"  模型类型: {summary['metadata'].get('model_type', 'N/A')}")
        print(f"  数据来源: {summary['metadata'].get('data_source', 'N/A')}")
        print(f"  状态: {summary['metadata'].get('status', 'unknown')}")
        
        print(f"\n时间信息:")
        print(f"  创建时间: {format_datetime(summary['metadata'].get('created_at', ''))}")
        if 'updated_at' in summary['metadata']:
            print(f"  更新时间: {format_datetime(summary['metadata']['updated_at'])}")
        
        if summary.get('model_exists'):
            print(f"\n模型信息:")
            print(f"  模型文件: {manager.get_model_path()}")
            print(f"  模型大小: {summary['model_size_mb']:.2f} MB")
        
        if 'checkpoints' in summary:
            print(f"\n检查点:")
            print(f"  数量: {summary['num_checkpoints']}")
            print(f"  列表: {', '.join(summary['checkpoints'][:5])}")
            if len(summary['checkpoints']) > 5:
                print(f"        ... 还有 {len(summary['checkpoints']) - 5} 个")
        
        if 'eval_results' in summary:
            print(f"\n评估结果:")
            results = summary['eval_results']
            print(f"  MAE: {results.get('mae', 0):.6f}")
            print(f"  RMSE: {results.get('rmse', 0):.6f}")
            print(f"  MAPE: {results.get('mape', 0):.2f}%")
            print(f"  R²: {results.get('r2', 0):.4f}")
            print(f"  方向准确率: {results.get('direction_accuracy', 0):.2f}%")
        
        # 显示配置
        config_path = manager.run_dir / "config.json"
        if config_path.exists():
            print(f"\n训练配置:")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            for key in ['seq_length', 'hidden_size', 'num_layers', 'epochs', 'batch_size', 'learning_rate']:
                if key in config:
                    print(f"  {key}: {config[key]}")
        
        # 显示文件列表
        print(f"\n输出文件:")
        files = [
            ("模型文件", manager.run_dir / "model" / "pytorch_model.bin"),
            ("模型配置", manager.run_dir / "model" / "config.json"),
            ("训练参数", manager.run_dir / "model" / "training_args.json"),
            ("训练配置", manager.run_dir / "config.json"),
            ("元数据", manager.run_dir / "metadata.json"),
            ("评估结果", manager.run_dir / "results" / "eval_results.json"),
            ("训练日志", manager.run_dir / "logs" / "training.log"),
            ("指标记录", manager.run_dir / "logs" / "metrics.json"),
        ]
        
        for name, path in files:
            if path.exists():
                size = path.stat().st_size
                if size > 1024 * 1024:
                    size_str = f"{size / (1024 * 1024):.2f} MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.2f} KB"
                else:
                    size_str = f"{size} B"
                print(f"  ✓ {name:15s}: {size_str:>10s}  {path}")
            else:
                print(f"  ✗ {name:15s}: 不存在")
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="查看训练运行")
    parser.add_argument("--run", type=str, help="显示指定运行的详情")
    parser.add_argument("--output-dir", type=str, default="out", help="输出目录")
    
    args = parser.parse_args()
    
    if args.run:
        show_run_details(args.run)
    else:
        list_runs()


if __name__ == "__main__":
    main()
