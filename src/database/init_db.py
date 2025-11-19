"""数据库初始化脚本"""
from src.database.connection import engine, Base
from src.database.models import (
    StockBasicInfo,
    StockKlineData,
    AIModel,
    AIPrediction,
    TechnicalIndicator
)


def create_tables():
    """创建所有表（仅创建AI相关的新表，不影响现有表）"""
    # 只创建AI相关的新表
    AIModel.__table__.create(engine, checkfirst=True)
    AIPrediction.__table__.create(engine, checkfirst=True)
    TechnicalIndicator.__table__.create(engine, checkfirst=True)
    print("AI相关表创建成功")


def drop_ai_tables():
    """删除AI相关表（谨慎使用）"""
    AIModel.__table__.drop(engine, checkfirst=True)
    AIPrediction.__table__.drop(engine, checkfirst=True)
    TechnicalIndicator.__table__.drop(engine, checkfirst=True)
    print("AI相关表已删除")


if __name__ == "__main__":
    create_tables()
