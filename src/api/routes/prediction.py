"""预测API路由"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime

from src.database.connection import get_db
from src.prediction.engine import PredictionEngine
from src.indicators.calculator import TechnicalIndicatorCalculator
from src.cache.redis_client import cache_manager
from src.api.schemas import PredictionRequest, PredictionResponse, PredictionItem

router = APIRouter(prefix="/api/v1/prediction", tags=["预测服务"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_stock(
    request: PredictionRequest,
    use_cache: bool = True,
    db: Session = Depends(get_db)
):
    """
    预测股票未来价格
    
    - **symbol**: 股票代码
    - **days**: 预测天数（默认5天）
    - **model_version**: 模型版本（可选）
    - **use_cache**: 是否使用缓存（默认True）
    """
    try:
        # 检查缓存
        if use_cache:
            cache_key = f"{request.symbol}:{request.model_version or 'default'}"
            cached_result = cache_manager.get_prediction(request.symbol, request.model_version or 'default')
            
            if cached_result:
                return PredictionResponse(**cached_result)
        
        # TODO: 从数据库加载模型配置
        # 这里使用默认配置作为示例
        model_config = {
            'input_size': 15,
            'hidden_size': 128,
            'num_layers': 2,
            'output_size': 1,
            'seq_length': 60
        }
        
        # 创建预测引擎
        # TODO: 实际应该从数据库或MinIO加载模型
        model_path = f"models/{request.symbol}_lstm_latest.pth"
        model_type = "lstm"
        
        engine = PredictionEngine(
            model_path=model_path,
            model_type=model_type,
            model_config=model_config
        )
        
        # 生成预测（带置信区间）
        result = engine.predict_with_confidence(
            symbol=request.symbol,
            days=request.days,
            confidence_level=0.95
        )
        
        # 构建响应
        predictions = [
            PredictionItem(
                date=pred['date'],
                price=pred['price'],
                confidence_lower=pred.get('confidence_lower'),
                confidence_upper=pred.get('confidence_upper')
            )
            for pred in result['predictions']
        ]
        
        response = PredictionResponse(
            symbol=result['symbol'],
            predictions=predictions,
            trend=result['trend'],
            confidence_score=result.get('confidence_score', 0.0)
        )
        
        # 缓存结果
        if use_cache:
            cache_manager.set_prediction(
                request.symbol,
                request.model_version or 'default',
                response.dict()
            )
        
        return response
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"未找到股票 {request.symbol} 的训练模型"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"预测失败: {str(e)}"
        )


@router.get("/indicators/{symbol}")
async def get_technical_indicators(
    symbol: str,
    date: Optional[str] = None,
    use_cache: bool = True,
    db: Session = Depends(get_db)
):
    """
    获取技术指标
    
    - **symbol**: 股票代码
    - **date**: 日期（可选，默认最新）
    - **use_cache**: 是否使用缓存
    """
    try:
        # 如果没有指定日期，使用最新日期
        if not date:
            from src.data.loader import StockDataLoader
            loader = StockDataLoader()
            date = loader.get_latest_trade_date(symbol)
        
        # 检查缓存
        if use_cache:
            cached_indicators = cache_manager.get_indicators(symbol, date)
            if cached_indicators:
                return cached_indicators
        
        # 计算技术指标
        calculator = TechnicalIndicatorCalculator(db_session=db)
        
        try:
            # 尝试从数据库获取
            indicators = calculator.get_latest_indicators(symbol)
            
            if not indicators:
                # 如果数据库没有，则计算并保存
                from datetime import datetime
                trade_date = datetime.strptime(date, '%Y-%m-%d').date()
                indicators = calculator.calculate_and_save(symbol, trade_date)
        except Exception as e:
            # 如果保存失败，至少返回计算结果
            df = calculator.calculate_all_indicators(symbol, end_date=date)
            if df.empty:
                raise HTTPException(status_code=404, detail=f"无法计算股票 {symbol} 的技术指标")
            
            latest_row = df.iloc[-1]
            indicators = {
                'symbol': symbol,
                'trade_date': date,
                'ma5': float(latest_row['ma5']) if not pd.isna(latest_row['ma5']) else None,
                'ma10': float(latest_row['ma10']) if not pd.isna(latest_row['ma10']) else None,
                'ma20': float(latest_row['ma20']) if not pd.isna(latest_row['ma20']) else None,
                'ma60': float(latest_row['ma60']) if not pd.isna(latest_row['ma60']) else None,
                'macd_dif': float(latest_row['macd_dif']) if not pd.isna(latest_row['macd_dif']) else None,
                'macd_dea': float(latest_row['macd_dea']) if not pd.isna(latest_row['macd_dea']) else None,
                'macd_bar': float(latest_row['macd_bar']) if not pd.isna(latest_row['macd_bar']) else None,
                'rsi6': float(latest_row['rsi6']) if not pd.isna(latest_row['rsi6']) else None,
                'rsi12': float(latest_row['rsi12']) if not pd.isna(latest_row['rsi12']) else None,
                'rsi24': float(latest_row['rsi24']) if not pd.isna(latest_row['rsi24']) else None,
                'boll_upper': float(latest_row['boll_upper']) if not pd.isna(latest_row['boll_upper']) else None,
                'boll_mid': float(latest_row['boll_mid']) if not pd.isna(latest_row['boll_mid']) else None,
                'boll_lower': float(latest_row['boll_lower']) if not pd.isna(latest_row['boll_lower']) else None,
            }
        
        # 缓存结果
        if use_cache:
            cache_manager.set_indicators(symbol, date, indicators)
        
        return indicators
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取技术指标失败: {str(e)}"
        )


@router.delete("/cache/{symbol}")
async def clear_prediction_cache(symbol: str):
    """
    清除指定股票的预测缓存
    
    - **symbol**: 股票代码
    """
    try:
        count = cache_manager.invalidate_stock_cache(symbol)
        return {
            "message": f"已清除股票 {symbol} 的缓存",
            "count": count
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"清除缓存失败: {str(e)}"
        )
