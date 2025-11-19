"""训练API路由"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
import uuid

from src.database.connection import get_db
from src.database.repositories import AIModelRepository
from src.api.schemas import TrainingRequest, TrainingResponse

router = APIRouter(prefix="/api/v1/training", tags=["训练服务"])


@router.post("/start", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    db: Session = Depends(get_db)
):
    """
    启动模型训练任务
    
    - **symbol**: 股票代码
    - **model_type**: 模型类型 (lstm/gru/transformer)
    - **config**: 训练配置（可选）
    """
    try:
        # 验证模型类型
        valid_types = ['lstm', 'gru', 'transformer']
        if request.model_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"无效的模型类型。支持的类型: {', '.join(valid_types)}"
            )
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 创建模型记录
        model_repo = AIModelRepository(db)
        
        # 默认配置
        default_config = {
            'input_size': 15,
            'hidden_size': 128,
            'num_layers': 2,
            'output_size': 1,
            'seq_length': 60,
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001
        }
        
        # 合并用户配置
        config = {**default_config, **(request.config or {})}
        
        # 创建模型记录
        model = model_repo.create(
            model_id=task_id,
            model_name=f"{request.symbol}_{request.model_type}",
            model_type=request.model_type,
            version="1.0",
            symbol=request.symbol,
            hyperparameters=config,
            status="training"
        )
        
        # 启动Celery异步训练任务
        from src.tasks.training import train_model_task
        train_model_task.delay(task_id, request.symbol, request.model_type, config)
        
        return TrainingResponse(
            task_id=task_id,
            status="started",
            message=f"训练任务已启动，任务ID: {task_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"启动训练失败: {str(e)}"
        )


@router.get("/status/{task_id}")
async def get_training_status(
    task_id: str,
    db: Session = Depends(get_db)
):
    """
    查询训练任务状态
    
    - **task_id**: 任务ID
    """
    try:
        model_repo = AIModelRepository(db)
        model = model_repo.get_by_model_id(task_id)
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"未找到任务 {task_id}"
            )
        
        response = {
            "task_id": model.model_id,
            "status": model.status,
            "model_name": model.model_name,
            "model_type": model.model_type,
            "symbol": model.symbol,
            "created_at": model.created_at.isoformat(),
            "updated_at": model.updated_at.isoformat()
        }
        
        # 如果有性能指标，添加到响应中
        if model.performance_metrics:
            response["metrics"] = model.performance_metrics
        
        # 如果训练完成，添加模型路径
        if model.status == "completed" and model.model_path:
            response["model_path"] = model.model_path
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"查询训练状态失败: {str(e)}"
        )


@router.delete("/cancel/{task_id}")
async def cancel_training(
    task_id: str,
    db: Session = Depends(get_db)
):
    """
    取消训练任务
    
    - **task_id**: 任务ID
    """
    try:
        model_repo = AIModelRepository(db)
        model = model_repo.get_by_model_id(task_id)
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"未找到任务 {task_id}"
            )
        
        if model.status not in ["training", "pending"]:
            raise HTTPException(
                status_code=400,
                detail=f"任务状态为 {model.status}，无法取消"
            )
        
        # 取消Celery任务
        from src.tasks.celery_app import celery_app
        celery_app.control.revoke(task_id, terminate=True)
        
        # 更新状态
        model_repo.update_status(task_id, "cancelled")
        
        return {
            "message": f"任务 {task_id} 已取消",
            "task_id": task_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"取消训练失败: {str(e)}"
        )
