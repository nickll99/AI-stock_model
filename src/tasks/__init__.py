"""Celery任务模块"""
from src.tasks.celery_app import celery_app
from src.tasks.training import train_model_task

__all__ = ['celery_app', 'train_model_task']
