"""
训练管理器 - 按照大模型行业标准组织训练输出
"""
import os
import json
import torch
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import logging

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingManager:
    """
    训练管理器 - 标准化训练输出结构
    
    输出目录结构（参考大模型标准）:
    out/
    ├── {run_name}/                    # 训练运行目录
    │   ├── config.json                # 训练配置
    │   ├── model/                     # 模型文件
    │   │   ├── pytorch_model.bin      # 模型权重
    │   │   ├── config.json            # 模型配置
    │   │   └── training_args.json     # 训练参数
    │   ├── checkpoints/               # 训练检查点
    │   │   ├── checkpoint-100/
    │   │   ├── checkpoint-200/
    │   │   └── ...
    │   ├── logs/                      # 训练日志
    │   │   ├── training.log           # 训练日志
    │   │   └── metrics.json           # 指标记录
    │   ├── results/                   # 评估结果
    │   │   ├── eval_results.json      # 评估指标
    │   │   └── predictions.json       # 预测结果
    │   └── metadata.json              # 元数据
    """
    
    def __init__(
        self,
        run_name: Optional[str] = None,
        output_dir: str = "out",
        stock_symbol: Optional[str] = None,
        model_type: str = "lstm"
    ):
        """
        初始化训练管理器
        
        Args:
            run_name: 训练运行名称，如果为None则自动生成
            output_dir: 输出根目录
            stock_symbol: 股票代码
            model_type: 模型类型
        """
        # 生成运行名称
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if stock_symbol:
                run_name = f"{stock_symbol}_{model_type}_{timestamp}"
            else:
                run_name = f"{model_type}_{timestamp}"
        
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.run_dir = self.output_dir / run_name
        self.stock_symbol = stock_symbol
        self.model_type = model_type
        
        # 创建目录结构
        self._create_directory_structure()
        
        # 初始化元数据
        self.metadata = {
            "run_name": run_name,
            "stock_symbol": stock_symbol,
            "model_type": model_type,
            "created_at": datetime.now().isoformat(),
            "status": "initialized",
            "data_source": "MySQL"
        }
        
        logger.info(f"训练管理器已初始化: {self.run_dir}")
    
    def _create_directory_structure(self):
        """创建标准目录结构"""
        directories = [
            self.run_dir,
            self.run_dir / "model",
            self.run_dir / "checkpoints",
            self.run_dir / "logs",
            self.run_dir / "results"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"目录结构已创建: {self.run_dir}")
    
    def save_config(self, config: Dict[str, Any]):
        """
        保存训练配置
        
        Args:
            config: 训练配置字典
        """
        config_path = self.run_dir / "config.json"
        
        # 添加元数据
        config_with_meta = {
            "run_name": self.run_name,
            "stock_symbol": self.stock_symbol,
            "model_type": self.model_type,
            "data_source": "MySQL",
            "created_at": datetime.now().isoformat(),
            **config
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_with_meta, f, indent=2, ensure_ascii=False)
        
        logger.info(f"配置已保存: {config_path}")
    
    def save_model(
        self,
        model: torch.nn.Module,
        model_config: Dict[str, Any],
        training_args: Dict[str, Any]
    ):
        """
        保存模型（最终模型）
        
        Args:
            model: PyTorch模型
            model_config: 模型配置
            training_args: 训练参数
        """
        model_dir = self.run_dir / "model"
        
        # 保存模型权重
        model_path = model_dir / "pytorch_model.bin"
        torch.save(model.state_dict(), model_path)
        logger.info(f"模型权重已保存: {model_path}")
        
        # 保存模型配置
        config_path = model_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2, ensure_ascii=False)
        logger.info(f"模型配置已保存: {config_path}")
        
        # 保存训练参数
        args_path = model_dir / "training_args.json"
        with open(args_path, 'w', encoding='utf-8') as f:
            json.dump(training_args, f, indent=2, ensure_ascii=False)
        logger.info(f"训练参数已保存: {args_path}")
        
        # 更新元数据
        self.metadata["model_saved"] = True
        self.metadata["model_path"] = str(model_path)
        self._save_metadata()
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """
        保存训练检查点
        
        Args:
            model: PyTorch模型
            optimizer: 优化器
            epoch: 当前epoch
            metrics: 评估指标
            is_best: 是否为最佳模型
        """
        checkpoint_dir = self.run_dir / "checkpoints" / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = checkpoint_dir / "pytorch_model.bin"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存指标
        metrics_path = checkpoint_dir / "metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"检查点已保存: {checkpoint_path}")
        
        # 如果是最佳模型，创建符号链接或复制
        if is_best:
            best_dir = self.run_dir / "checkpoints" / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(checkpoint_dir, best_dir)
            logger.info(f"最佳模型已保存: {best_dir}")
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float], phase: str = "train"):
        """
        记录训练指标
        
        Args:
            epoch: 当前epoch
            metrics: 指标字典
            phase: 阶段（train/val/test）
        """
        log_file = self.run_dir / "logs" / "metrics.json"
        
        # 读取现有日志
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # 添加新记录
        log_entry = {
            "epoch": epoch,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        logs.append(log_entry)
        
        # 保存日志
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    
    def save_evaluation_results(self, results: Dict[str, Any]):
        """
        保存评估结果
        
        Args:
            results: 评估结果字典
        """
        results_path = self.run_dir / "results" / "eval_results.json"
        
        results_with_meta = {
            "evaluated_at": datetime.now().isoformat(),
            "stock_symbol": self.stock_symbol,
            "model_type": self.model_type,
            **results
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_with_meta, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估结果已保存: {results_path}")
        
        # 更新元数据
        self.metadata["evaluation_completed"] = True
        self.metadata["eval_metrics"] = results
        self._save_metadata()
    
    def save_predictions(self, predictions: Dict[str, Any]):
        """
        保存预测结果
        
        Args:
            predictions: 预测结果字典
        """
        pred_path = self.run_dir / "results" / "predictions.json"
        
        with open(pred_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"预测结果已保存: {pred_path}")
    
    def save_training_log(self, message: str):
        """
        保存训练日志
        
        Args:
            message: 日志消息
        """
        log_file = self.run_dir / "logs" / "training.log"
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def _save_metadata(self):
        """保存元数据"""
        metadata_path = self.run_dir / "metadata.json"
        
        self.metadata["updated_at"] = datetime.now().isoformat()
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def update_status(self, status: str, **kwargs):
        """
        更新训练状态
        
        Args:
            status: 状态（training/completed/failed）
            **kwargs: 其他元数据
        """
        self.metadata["status"] = status
        self.metadata.update(kwargs)
        self._save_metadata()
        
        logger.info(f"状态已更新: {status}")
    
    def get_model_path(self) -> Path:
        """获取模型路径"""
        return self.run_dir / "model" / "pytorch_model.bin"
    
    def get_best_checkpoint_path(self) -> Path:
        """获取最佳检查点路径"""
        return self.run_dir / "checkpoints" / "best" / "pytorch_model.bin"
    
    def load_model_config(self) -> Dict[str, Any]:
        """加载模型配置"""
        config_path = self.run_dir / "model" / "config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"模型配置不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取训练摘要
        
        Returns:
            训练摘要字典
        """
        summary = {
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "metadata": self.metadata
        }
        
        # 添加模型信息
        model_path = self.get_model_path()
        if model_path.exists():
            summary["model_exists"] = True
            summary["model_size_mb"] = model_path.stat().st_size / (1024 * 1024)
        
        # 添加检查点信息
        checkpoints_dir = self.run_dir / "checkpoints"
        if checkpoints_dir.exists():
            checkpoints = [d.name for d in checkpoints_dir.iterdir() if d.is_dir()]
            summary["checkpoints"] = checkpoints
            summary["num_checkpoints"] = len(checkpoints)
        
        # 添加评估结果
        eval_path = self.run_dir / "results" / "eval_results.json"
        if eval_path.exists():
            with open(eval_path, 'r', encoding='utf-8') as f:
                summary["eval_results"] = json.load(f)
        
        return summary
    
    @staticmethod
    def list_runs(output_dir: str = "out") -> list:
        """
        列出所有训练运行
        
        Args:
            output_dir: 输出目录
            
        Returns:
            运行列表
        """
        output_path = Path(output_dir)
        
        if not output_path.exists():
            return []
        
        runs = []
        for run_dir in output_path.iterdir():
            if run_dir.is_dir():
                metadata_path = run_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    runs.append({
                        "run_name": run_dir.name,
                        "run_dir": str(run_dir),
                        **metadata
                    })
        
        # 按创建时间排序
        runs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return runs
    
    @staticmethod
    def load_run(run_name: str, output_dir: str = "out") -> 'TrainingManager':
        """
        加载已有的训练运行
        
        Args:
            run_name: 运行名称
            output_dir: 输出目录
            
        Returns:
            TrainingManager实例
        """
        run_dir = Path(output_dir) / run_name
        
        if not run_dir.exists():
            raise FileNotFoundError(f"训练运行不存在: {run_dir}")
        
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"元数据不存在: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        manager = TrainingManager(
            run_name=run_name,
            output_dir=output_dir,
            stock_symbol=metadata.get("stock_symbol"),
            model_type=metadata.get("model_type", "lstm")
        )
        
        manager.metadata = metadata
        
        logger.info(f"训练运行已加载: {run_dir}")
        
        return manager
