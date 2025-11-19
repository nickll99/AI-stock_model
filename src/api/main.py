"""FastAPI主应用"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from src.config import settings
from src.api.routes import data, prediction, training, models
from src.api.middleware import (
    ErrorHandlerMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware
)
from src.utils.logger import setup_logging

# 配置日志系统
setup_logging(
    log_level=settings.LOG_LEVEL,
    log_file=settings.LOG_FILE if hasattr(settings, 'LOG_FILE') else "logs/app.log",
    json_format=True
)

# API文档描述
API_DESCRIPTION = """
## A股AI预测系统API

基于深度学习的A股市场智能分析和预测平台，提供以下核心功能：

### 主要功能模块

* **数据服务** - 获取股票基本信息和历史K线数据
* **预测服务** - 使用AI模型生成股票价格预测和技术指标
* **训练服务** - 训练和管理深度学习模型（LSTM、GRU、Transformer）
* **模型管理** - 查询和管理已训练的模型版本

### 技术特性

* 支持多种深度学习模型（LSTM、GRU、Transformer）
* 提供95%置信区间的价格预测
* 实时计算技术指标（MA、MACD、RSI、布林带等）
* Redis缓存加速响应
* 异步任务处理（Celery）

### 认证和限流

* API限流：100次/分钟（基于IP）
* 所有响应包含请求追踪ID（X-Request-ID）
* 结构化JSON日志记录

### 错误处理

所有错误响应遵循统一格式：
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "错误描述",
    "details": {},
    "timestamp": "2025-11-18T10:30:00Z",
    "request_id": "uuid"
  }
}
```
"""

app = FastAPI(
    title="A股AI预测系统",
    description=API_DESCRIPTION,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "数据服务",
            "description": "股票数据查询接口，包括基本信息和K线数据"
        },
        {
            "name": "预测服务",
            "description": "AI预测接口，生成价格预测和技术指标"
        },
        {
            "name": "训练服务",
            "description": "模型训练接口，支持异步训练任务"
        },
        {
            "name": "模型管理",
            "description": "模型查询和管理接口"
        },
        {
            "name": "系统",
            "description": "系统健康检查和状态接口"
        }
    ],
    contact={
        "name": "A股AI预测系统",
        "email": "support@example.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)


def custom_openapi():
    """自定义OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags
    )
    
    # 添加自定义响应头说明
    openapi_schema["components"]["headers"] = {
        "X-Request-ID": {
            "description": "请求追踪ID，用于日志关联和问题排查",
            "schema": {"type": "string"}
        },
        "X-Process-Time": {
            "description": "请求处理时间（秒）",
            "schema": {"type": "string"}
        },
        "X-RateLimit-Limit": {
            "description": "速率限制 - 时间窗口内允许的最大请求数",
            "schema": {"type": "integer"}
        },
        "X-RateLimit-Remaining": {
            "description": "速率限制 - 当前时间窗口内剩余请求数",
            "schema": {"type": "integer"}
        },
        "X-RateLimit-Reset": {
            "description": "速率限制 - 重置时间戳（Unix时间）",
            "schema": {"type": "integer"}
        }
    }
    
    # 添加通用错误响应模型
    openapi_schema["components"]["schemas"]["ErrorResponse"] = {
        "type": "object",
        "properties": {
            "error": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "错误代码"},
                    "message": {"type": "string", "description": "错误描述"},
                    "details": {"type": "object", "description": "详细错误信息"},
                    "timestamp": {"type": "string", "format": "date-time", "description": "错误发生时间"},
                    "request_id": {"type": "string", "description": "请求追踪ID"}
                }
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# 添加中间件（注意顺序：后添加的先执行）
# 1. 请求日志中间件
app.add_middleware(RequestLoggingMiddleware)

# 2. 限流中间件 (100次/分钟)
app.add_middleware(RateLimitMiddleware, calls=100, period=60)

# 3. 错误处理中间件
app.add_middleware(ErrorHandlerMiddleware)

# 4. CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(data.router)
app.include_router(prediction.router)
app.include_router(training.router)
app.include_router(models.router)


@app.get(
    "/",
    tags=["系统"],
    summary="API根路径",
    description="返回API基本信息和文档链接"
)
async def root():
    """
    API根路径
    
    返回系统基本信息和文档链接
    """
    return {
        "message": "A股AI预测系统API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "status": "running"
    }


@app.get(
    "/health",
    tags=["系统"],
    summary="健康检查",
    description="检查服务运行状态",
    responses={
        200: {
            "description": "服务正常运行",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "service": "A股AI预测系统",
                        "version": "1.0.0"
                    }
                }
            }
        }
    }
)
async def health_check():
    """
    健康检查
    
    用于监控系统运行状态，返回服务健康信息
    """
    return {
        "status": "healthy",
        "service": "A股AI预测系统",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
