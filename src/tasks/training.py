"""模型训练Celery任务"""
from celery import Task
from datetime import datetime, date
import torch
import logging
from typing import Dict

from src.tasks.celery_app import celery_app
from src.database.connection import get_db_context
from src.database.repositories import AIModelRepository
from src.data.loader import StockDataLoader
from src.features.dataset_builder import FeatureDatasetBuilder
from src.training.trainer import ModelTrainer
from src.training.evaluator import ModelEvaluator
from src.models.lstm_model import LSTMModel
from src.models.gru_model import GRUModel
from src.models.transformer_model import TransformerModel
from src.storage.minio_client import MinIOClient

logger = logging.getLogger(__name__)


class TrainingTask(Task):
    """训练任务基类"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """任务失败回调"""
        logger.error(f"训练任务失败 [Task-ID: {task_id}]: {exc}")
        
        # 更新数据库状态
        try:
            with get_db_context() as db:
                repo = AIModelRepository(db)
                repo.update_status(task_id, 'failed')
        except Exception as e:
            logger.error(f"更新任务状态失败: {e}")
    
    def on_success(self, retval, task_id, args, kwargs):
        """任务成功回调"""
        logger.info(f"训练任务成功 [Task-ID: {task_id}]")


@celery_app.task(
    base=TrainingTask,
    bind=True,
    max_retries=3,
    default_retry_delay=300  # 5分钟后重试
)
def train_model_task(
    self,
    model_id: str,
    symbol: str,
    model_type: str,
    config: Dict
):
    """
    模型训练任务
    
    Args:
        self: Celery任务实例
        model_id: 模型ID
        symbol: 股票代码
        model_type: 模型类型
        config: 训练配置
    """
    try:
        logger.info(f"开始训练任务 [Model-ID: {model_id}] {symbol} - {model_type}")
        
        # 1. 加载数据
        logger.info("加载训练数据...")
        data_loader = StockDataLoader()
        
        # 获取3年历史数据
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now().replace(year=datetime.now().year - 3)).strftime('%Y-%m-%d')
        
        df = data_loader.load_kline_data(symbol, start_date, end_date)
        
        if df.empty or len(df) < config.get('seq_length', 60) + 100:
            raise ValueError(f"数据不足: 需要至少 {config.get('seq_length', 60) + 100} 条记录")
        
        # 2. 构建特征数据集
        logger.info("构建特征数据集...")
        dataset_builder = FeatureDatasetBuilder()
        
        train_loader, val_loader, test_loader = dataset_builder.build_dataloaders(
            df=df,
            seq_length=config.get('seq_length', 60),
            batch_size=config.get('batch_size', 32),
            train_ratio=0.7,
            val_ratio=0.15
        )
        
        # 3. 创建模型
        logger.info(f"创建{model_type}模型...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if model_type == 'lstm':
            model = LSTMModel(
                input_size=config.get('input_size', 15),
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 2),
                output_size=config.get('output_size', 1),
                dropout=config.get('dropout', 0.2)
            )
        elif model_type == 'gru':
            model = GRUModel(
                input_size=config.get('input_size', 15),
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 2),
                output_size=config.get('output_size', 1),
                dropout=config.get('dropout', 0.2)
            )
        elif model_type == 'transformer':
            model = TransformerModel(
                input_size=config.get('input_size', 15),
                d_model=config.get('d_model', 128),
                nhead=config.get('nhead', 8),
                num_layers=config.get('num_layers', 3),
                output_size=config.get('output_size', 1),
                dropout=config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 4. 训练模型
        logger.info("开始训练...")
        trainer = ModelTrainer(
            model=model,
            device=device,
            learning_rate=config.get('learning_rate', 0.001)
        )
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.get('epochs', 100),
            patience=config.get('patience', 10),
            checkpoint_dir=f'checkpoints/{model_id}'
        )
        
        # 5. 评估模型
        logger.info("评估模型...")
        evaluator = ModelEvaluator(model, device=device)
        
        # 准备测试数据
        X_test, y_test = [], []
        for batch_X, batch_y in test_loader:
            X_test.append(batch_X.numpy())
            y_test.append(batch_y.numpy())
        
        import numpy as np
        X_test = np.concatenate(X_test)
        y_test = np.concatenate(y_test)
        
        metrics = evaluator.evaluate(X_test, y_test)
        
        logger.info(f"评估结果: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        # 6. 保存模型到MinIO
        logger.info("保存模型...")
        model_filename = f"{symbol}_{model_type}_{model_id}.pth"
        local_path = f"checkpoints/{model_id}/best_model.pth"
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'metrics': metrics,
            'history': history
        }, local_path)
        
        # 上传到MinIO
        minio_client = MinIOClient()
        minio_path = minio_client.upload_model(local_path, model_filename)
        
        # 7. 更新数据库
        logger.info("更新数据库...")
        with get_db_context() as db:
            repo = AIModelRepository(db)
            repo.update(
                model_id=model_id,
                status='completed',
                training_start_date=datetime.strptime(start_date, '%Y-%m-%d').date(),
                training_end_date=datetime.strptime(end_date, '%Y-%m-%d').date(),
                performance_metrics=metrics,
                model_path=minio_path
            )
        
        logger.info(f"训练任务完成 [Model-ID: {model_id}]")
        
        return {
            'model_id': model_id,
            'status': 'completed',
            'metrics': metrics,
            'model_path': minio_path
        }
        
    except Exception as exc:
        logger.error(f"训练任务异常 [Model-ID: {model_id}]: {exc}", exc_info=True)
        
        # 更新状态为失败
        try:
            with get_db_context() as db:
                repo = AIModelRepository(db)
                repo.update_status(model_id, 'failed')
        except Exception as e:
            logger.error(f"更新失败状态异常: {e}")
        
        # 重试任务
        raise self.retry(exc=exc)


@celery_app.task
def cleanup_old_models_task():
    """清理旧模型任务（定期任务）"""
    logger.info("开始清理旧模型...")
    
    try:
        with get_db_context() as db:
            repo = AIModelRepository(db)
            
            # 获取所有失败的模型
            failed_models = repo.list_models(filters={'status': 'failed'}, limit=1000)
            
            # 删除超过7天的失败模型
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=7)
            
            deleted_count = 0
            for model in failed_models:
                if model.created_at < cutoff_date:
                    repo.delete(model.model_id)
                    deleted_count += 1
            
            logger.info(f"清理完成: 删除了 {deleted_count} 个旧模型")
            
            return {'deleted_count': deleted_count}
            
    except Exception as exc:
        logger.error(f"清理旧模型失败: {exc}", exc_info=True)
        raise
