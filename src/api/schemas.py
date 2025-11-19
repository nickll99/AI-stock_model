"""API数据模型"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import date, datetime
import re


# 请求模型
class PredictionRequest(BaseModel):
    symbol: str = Field(..., min_length=6, max_length=6, description="股票代码(6位)")
    days: int = Field(5, ge=1, le=30, description="预测天数(1-30)")
    model_version: Optional[str] = Field(None, max_length=50, description="模型版本")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """验证股票代码格式"""
        if not re.match(r'^\d{6}$', v):
            raise ValueError('股票代码必须是6位数字')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "000001",
                "days": 5,
                "model_version": "v1.0"
            }
        }


class TrainingRequest(BaseModel):
    symbol: str = Field(..., min_length=6, max_length=6, description="股票代码(6位)")
    model_type: str = Field(..., description="模型类型")
    config: Optional[Dict] = Field(None, description="训练配置")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """验证股票代码格式"""
        if not re.match(r'^\d{6}$', v):
            raise ValueError('股票代码必须是6位数字')
        return v
    
    @validator('model_type')
    def validate_model_type(cls, v):
        """验证模型类型"""
        valid_types = ['lstm', 'gru', 'transformer']
        if v.lower() not in valid_types:
            raise ValueError(f'模型类型必须是以下之一: {", ".join(valid_types)}')
        return v.lower()
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "000001",
                "model_type": "lstm",
                "config": {
                    "epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 0.001
                }
            }
        }


# 响应模型
class StockInfo(BaseModel):
    symbol: str
    name: Optional[str] = None


class StockListResponse(BaseModel):
    stocks: List[StockInfo]
    count: int


class KlineDataResponse(BaseModel):
    symbol: str
    data: List[Dict]
    count: int


class PredictionItem(BaseModel):
    date: str
    price: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None


class PredictionResponse(BaseModel):
    symbol: str
    predictions: List[PredictionItem]
    trend: str
    confidence_score: float


class ModelInfo(BaseModel):
    model_id: str
    model_name: str
    model_type: str
    version: str
    symbol: Optional[str]
    status: str
    performance_metrics: Optional[Dict]
    created_at: str


class ModelListResponse(BaseModel):
    models: List[ModelInfo]
    count: int


class TrainingResponse(BaseModel):
    task_id: str
    status: str
    message: str



# 错误响应模型
class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict] = None
    timestamp: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
    
    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "请求参数验证失败",
                    "details": {
                        "field": "symbol",
                        "error": "股票代码必须是6位数字"
                    },
                    "timestamp": "2025-11-12T10:30:00Z"
                }
            }
        }
