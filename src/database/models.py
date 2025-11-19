from sqlalchemy import Column, BigInteger, String, Integer, Date, DateTime, DECIMAL, Text, JSON
from sqlalchemy.sql import func
from src.database.connection import Base


class StockBasicInfo(Base):
    """股票基本信息表"""
    __tablename__ = "stock_basic_info"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="主键ID")
    symbol = Column(String(10), unique=True, nullable=False, index=True, comment="股票代码")
    name = Column(String(50), nullable=False, comment="股票名称")
    exchange = Column(String(10), comment="交易所")
    market = Column(String(10), nullable=False, index=True, comment="交易市场")
    stock_type = Column(String(10), nullable=False, comment="股票类型")
    industry = Column(String(30), index=True, comment="所属行业(证监会)")
    list_date = Column(Date, comment="上市日期")
    delist_date = Column(Date, comment="退市日期")
    is_active = Column(Integer, nullable=False, index=True, comment="是否上市交易")
    total_share = Column(DECIMAL(20, 2), comment="总股本(股)")
    float_share = Column(DECIMAL(20, 2), comment="流通股本(股)")
    total_market_value = Column(DECIMAL(20, 2), comment="总市值(元)")
    float_market_value = Column(DECIMAL(20, 2), comment="流通市值(元)")
    main_business = Column(Text, comment="主营业务")
    business_scope = Column(Text, comment="经营范围")
    company_profile = Column(Text, comment="公司简介")
    registered_address = Column(String(200), comment="注册地址")
    office_address = Column(String(200), comment="办公地址")
    legal_person = Column(String(50), comment="法人代表")
    president = Column(String(50), comment="总经理")
    secretary = Column(String(50), comment="董秘")
    chairman = Column(String(50), comment="董事长")
    province = Column(String(50), comment="所在省份")
    city = Column(String(50), comment="所在城市")
    reg_capital = Column(DECIMAL(20, 4), comment="注册资本(万元)")
    employee_count = Column(Integer, comment="员工人数")
    website = Column(String(200), comment="公司网站")
    phone = Column(String(100), comment="联系电话")
    email = Column(String(100), comment="电子邮箱")
    created_at = Column(DateTime, nullable=False, server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now(), comment="更新时间")


class StockKlineData(Base):
    """日线数据表"""
    __tablename__ = "stock_kline_data"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True, comment="股票代码")
    trade_date = Column(Date, nullable=False, index=True, comment="交易日期")
    open = Column(DECIMAL(10, 3), nullable=False, comment="开盘价")
    high = Column(DECIMAL(10, 3), nullable=False, comment="最高价")
    low = Column(DECIMAL(10, 3), nullable=False, comment="最低价")
    close = Column(DECIMAL(10, 3), nullable=False, comment="收盘价")
    pre_close = Column(DECIMAL(10, 3), comment="昨收价")
    change = Column(DECIMAL(10, 3), comment="涨跌额")
    pct_chg = Column(DECIMAL(8, 3), index=True, comment="涨跌幅(%)")
    vol = Column(BigInteger, comment="成交量(手)")
    amount = Column(DECIMAL(15, 2), comment="成交额(千元)")
    turnover_rate = Column(DECIMAL(8, 3), comment="换手率(%)")
    pe = Column(DECIMAL(10, 3), comment="市盈率")
    pb = Column(DECIMAL(10, 3), comment="市净率")
    total_mv = Column(DECIMAL(15, 2), comment="总市值(万元)")
    circ_mv = Column(DECIMAL(15, 2), comment="流通市值(万元)")
    limit_status = Column(Integer, comment="涨跌停状态 0正常 1涨停 -1跌停")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class AIModel(Base):
    """AI模型信息表"""
    __tablename__ = "ai_models"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    model_id = Column(String(36), unique=True, nullable=False, index=True, comment="模型UUID")
    model_name = Column(String(100), nullable=False, comment="模型名称")
    model_type = Column(String(50), nullable=False, index=True, comment="模型类型: lstm, gru, transformer")
    version = Column(String(20), nullable=False, comment="版本号")
    symbol = Column(String(10), index=True, comment="股票代码(NULL表示通用模型)")
    training_start_date = Column(Date, comment="训练数据起始日期")
    training_end_date = Column(Date, comment="训练数据结束日期")
    hyperparameters = Column(JSON, comment="超参数配置")
    performance_metrics = Column(JSON, comment="性能指标")
    model_path = Column(String(255), comment="模型文件路径")
    status = Column(String(20), nullable=False, index=True, comment="状态: training, completed, failed")
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())


class AIPrediction(Base):
    """AI预测结果表"""
    __tablename__ = "ai_predictions"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    model_id = Column(String(36), nullable=False, index=True, comment="模型ID")
    symbol = Column(String(10), nullable=False, index=True, comment="股票代码")
    prediction_date = Column(Date, nullable=False, index=True, comment="预测生成日期")
    target_date = Column(Date, nullable=False, index=True, comment="预测目标日期")
    predicted_close = Column(DECIMAL(10, 3), nullable=False, comment="预测收盘价")
    confidence_lower = Column(DECIMAL(10, 3), comment="置信区间下限")
    confidence_upper = Column(DECIMAL(10, 3), comment="置信区间上限")
    actual_close = Column(DECIMAL(10, 3), comment="实际收盘价")
    prediction_error = Column(DECIMAL(10, 3), comment="预测误差")
    created_at = Column(DateTime, nullable=False, server_default=func.now())


class TechnicalIndicator(Base):
    """技术指标缓存表"""
    __tablename__ = "technical_indicators"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True, comment="股票代码")
    trade_date = Column(Date, nullable=False, index=True, comment="交易日期")
    ma5 = Column(DECIMAL(10, 3), comment="5日均线")
    ma10 = Column(DECIMAL(10, 3), comment="10日均线")
    ma20 = Column(DECIMAL(10, 3), comment="20日均线")
    ma60 = Column(DECIMAL(10, 3), comment="60日均线")
    ema12 = Column(DECIMAL(10, 3), comment="12日指数移动平均")
    ema26 = Column(DECIMAL(10, 3), comment="26日指数移动平均")
    macd_dif = Column(DECIMAL(10, 3), comment="MACD DIF")
    macd_dea = Column(DECIMAL(10, 3), comment="MACD DEA")
    macd_bar = Column(DECIMAL(10, 3), comment="MACD柱")
    rsi6 = Column(DECIMAL(10, 3), comment="6日RSI")
    rsi12 = Column(DECIMAL(10, 3), comment="12日RSI")
    rsi24 = Column(DECIMAL(10, 3), comment="24日RSI")
    boll_upper = Column(DECIMAL(10, 3), comment="布林带上轨")
    boll_mid = Column(DECIMAL(10, 3), comment="布林带中轨")
    boll_lower = Column(DECIMAL(10, 3), comment="布林带下轨")
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
