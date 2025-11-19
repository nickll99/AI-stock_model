"""Celery应用配置"""
from celery import Celery
from src.config import settings

# 创建Celery应用
celery_app = Celery(
    'stock_ai_tasks',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=['src.tasks.training']
)

# Celery配置
celery_app.conf.update(
    # 任务结果过期时间(秒)
    result_expires=3600,
    
    # 任务序列化方式
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    
    # 时区设置
    timezone='Asia/Shanghai',
    enable_utc=True,
    
    # 任务路由
    task_routes={
        'src.tasks.training.*': {'queue': 'training'},
    },
    
    # 任务限流
    task_annotations={
        'src.tasks.training.train_model_task': {
            'rate_limit': '10/m'  # 每分钟最多10个训练任务
        }
    },
    
    # Worker配置
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    
    # 任务失败重试
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# 任务自动发现
celery_app.autodiscover_tasks(['src.tasks'])
