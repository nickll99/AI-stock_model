"""模型管理API路由"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from src.database.connection import get_db
from src.database.repositories import AIModelRepository
from src.api.schemas import ModelListResponse, ModelInfo
from src.cache.redis_client import cache_manager

router = APIRouter(prefix="/api/v1/models", tags=["模型管理"])


@router.get("/list", response_model=ModelListResponse)
async def list_models(
    symbol: Optional[str] = Query(None, description="股票代码筛选"),
    model_type: Optional[str] = Query(None, description="模型类型筛选"),
    status: Optional[str] = Query(None, description="状态筛选"),
    limit: int = Query(20, ge=1, le=100, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量"),
    db: Session = Depends(get_db)
):
    """
    获取模型列表
    
    - **symbol**: 股票代码筛选（可选）
    - **model_type**: 模型类型筛选（可选）
    - **status**: 状态筛选（可选）
    - **limit**: 返回数量限制
    - **offset**: 分页偏移量
    """
    try:
        model_repo = AIModelRepository(db)
        
        # 构建查询条件
        filters = {}
        if symbol:
            filters['symbol'] = symbol
        if model_type:
            filters['model_type'] = model_type
        if status:
            filters['status'] = status
        
        # 查询模型
        models = model_repo.list_models(
            filters=filters,
            limit=limit,
            offset=offset
        )
        
        # 转换为响应格式
        model_infos = [
            ModelInfo(
                model_id=model.model_id,
                model_name=model.model_name,
                model_type=model.model_type,
                version=model.version,
                symbol=model.symbol,
                status=model.status,
                performance_metrics=model.performance_metrics,
                created_at=model.created_at.isoformat()
            )
            for model in models
        ]
        
        return ModelListResponse(
            models=model_infos,
            count=len(model_infos)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取模型列表失败: {str(e)}"
        )


@router.get("/{model_id}")
async def get_model_detail(
    model_id: str,
    use_cache: bool = True,
    db: Session = Depends(get_db)
):
    """
    获取模型详细信息
    
    - **model_id**: 模型ID
    - **use_cache**: 是否使用缓存
    """
    try:
        # 检查缓存
        if use_cache:
            cached_metadata = cache_manager.get_model_metadata(model_id)
            if cached_metadata:
                return cached_metadata
        
        model_repo = AIModelRepository(db)
        model = model_repo.get_by_model_id(model_id)
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"未找到模型 {model_id}"
            )
        
        # 构建详细信息
        detail = {
            "model_id": model.model_id,
            "model_name": model.model_name,
            "model_type": model.model_type,
            "version": model.version,
            "symbol": model.symbol,
            "status": model.status,
            "training_start_date": model.training_start_date.isoformat() if model.training_start_date else None,
            "training_end_date": model.training_end_date.isoformat() if model.training_end_date else None,
            "hyperparameters": model.hyperparameters,
            "performance_metrics": model.performance_metrics,
            "model_path": model.model_path,
            "created_at": model.created_at.isoformat(),
            "updated_at": model.updated_at.isoformat()
        }
        
        # 缓存结果
        if use_cache:
            cache_manager.set_model_metadata(model_id, detail)
        
        return detail
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取模型详情失败: {str(e)}"
        )


@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    db: Session = Depends(get_db)
):
    """
    删除模型
    
    - **model_id**: 模型ID
    """
    try:
        model_repo = AIModelRepository(db)
        model = model_repo.get_by_model_id(model_id)
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"未找到模型 {model_id}"
            )
        
        # TODO: 同时删除MinIO中的模型文件
        # if model.model_path:
        #     from src.storage.minio_client import MinIOClient
        #     minio_client = MinIOClient()
        #     minio_client.delete_file(model.model_path)
        
        # 删除数据库记录
        model_repo.delete(model_id)
        
        # 清除缓存
        cache_manager.delete_model_metadata(model_id)
        
        return {
            "message": f"模型 {model_id} 已删除",
            "model_id": model_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"删除模型失败: {str(e)}"
        )


@router.get("/{model_id}/download")
async def download_model(
    model_id: str,
    db: Session = Depends(get_db)
):
    """
    下载模型文件
    
    - **model_id**: 模型ID
    """
    try:
        model_repo = AIModelRepository(db)
        model = model_repo.get_by_model_id(model_id)
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"未找到模型 {model_id}"
            )
        
        if not model.model_path:
            raise HTTPException(
                status_code=404,
                detail=f"模型 {model_id} 没有关联的文件"
            )
        
        # TODO: 从MinIO获取下载URL
        # from src.storage.minio_client import MinIOClient
        # minio_client = MinIOClient()
        # download_url = minio_client.get_presigned_url(model.model_path)
        
        return {
            "model_id": model_id,
            "model_path": model.model_path,
            "message": "模型下载功能待实现"
            # "download_url": download_url
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取下载链接失败: {str(e)}"
        )
