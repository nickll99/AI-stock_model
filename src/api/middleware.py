"""API中间件"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime
import time
import uuid
import logging
from typing import Dict
from collections import defaultdict

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """统一错误处理中间件"""
    
    async def dispatch(self, request: Request, call_next):
        """处理请求并捕获异常"""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
            
        except RequestValidationError as exc:
            # Pydantic验证错误
            logger.warning(f"验证错误 [Request-ID: {request_id}]: {exc}")
            
            errors = []
            for error in exc.errors():
                errors.append({
                    "field": ".".join(str(x) for x in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"]
                })
            
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "请求参数验证失败",
                        "details": errors,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "request_id": request_id
                    }
                }
            )
            
        except ValueError as exc:
            # 值错误
            logger.warning(f"值错误 [Request-ID: {request_id}]: {exc}")
            
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": {
                        "code": "VALUE_ERROR",
                        "message": str(exc),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "request_id": request_id
                    }
                }
            )
            
        except Exception as exc:
            # 未预期的错误
            logger.error(f"服务器错误 [Request-ID: {request_id}]: {exc}", exc_info=True)
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": "服务器内部错误",
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "request_id": request_id
                    }
                }
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """API限流中间件 - 基于IP的简单限流"""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        """
        初始化限流中间件
        
        Args:
            app: FastAPI应用
            calls: 时间窗口内允许的请求次数
            period: 时间窗口(秒)
        """
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.requests: Dict[str, list] = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        """处理请求并检查限流"""
        # 获取客户端IP
        client_ip = request.client.host if request.client else "unknown"
        
        # 当前时间
        now = time.time()
        
        # 清理过期的请求记录
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < self.period
        ]
        
        # 检查是否超过限流
        if len(self.requests[client_ip]) >= self.calls:
            logger.warning(f"限流触发: IP {client_ip} 超过 {self.calls} 次/{self.period}秒")
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"请求过于频繁，请在 {self.period} 秒后重试",
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(self.calls),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(now + self.period))
                }
            )
        
        # 记录本次请求
        self.requests[client_ip].append(now)
        
        # 继续处理请求
        response = await call_next(request)
        
        # 添加限流信息到响应头
        remaining = self.calls - len(self.requests[client_ip])
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(now + self.period))
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件 - 记录所有请求和响应，添加追踪ID"""
    
    async def dispatch(self, request: Request, call_next):
        """记录请求和响应信息"""
        # 生成或获取请求ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        
        # 记录请求开始时间
        start_time = time.time()
        
        # 获取客户端信息
        client_ip = request.client.host if request.client else "unknown"
        
        # 创建结构化日志记录
        log_extra = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_ip": client_ip,
            "user_agent": request.headers.get("user-agent", "unknown")
        }
        
        # 记录请求开始
        logger.info(
            f"请求开始: {request.method} {request.url.path}",
            extra=log_extra
        )
        
        # 处理请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 更新日志信息
        log_extra.update({
            "status_code": response.status_code,
            "process_time": f"{process_time:.3f}s"
        })
        
        # 记录响应信息
        log_level = logging.INFO if response.status_code < 400 else logging.WARNING
        logger.log(
            log_level,
            f"请求完成: {request.method} {request.url.path} - "
            f"状态码: {response.status_code} - 耗时: {process_time:.3f}s",
            extra=log_extra
        )
        
        # 添加追踪信息到响应头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        
        return response
