"""数据查询API路由"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List

from src.database.connection import get_db
from src.data.loader import StockDataLoader
from src.api.schemas import StockListResponse, KlineDataResponse, StockInfo

router = APIRouter(prefix="/api/v1/data", tags=["数据服务"])


@router.get(
    "/stocks",
    response_model=StockListResponse,
    summary="获取股票列表",
    description="查询A股市场的股票列表，支持按市场、行业筛选和分页",
    responses={
        200: {"description": "成功返回股票列表"},
        500: {"description": "服务器内部错误"}
    }
)
async def get_stocks(
    market: Optional[str] = Query(None, description="交易市场筛选（如：主板、创业板）"),
    industry: Optional[str] = Query(None, description="所属行业筛选（如：银行、科技）"),
    is_active: Optional[int] = Query(1, description="是否上市交易（1=是，0=否）"),
    limit: int = Query(100, ge=1, le=1000, description="返回数量限制，最大1000"),
    offset: int = Query(0, ge=0, description="分页偏移量"),
    db: Session = Depends(get_db)
):
    """
    获取股票列表
    
    查询A股市场的股票列表，支持多种筛选条件：
    
    - **market**: 交易市场筛选（如：主板、创业板、科创板）
    - **industry**: 行业筛选（如：银行、科技、医药）
    - **is_active**: 是否上市交易（1=是，0=否）
    - **limit**: 返回数量限制（1-1000）
    - **offset**: 分页偏移量
    
    **示例请求：**
    ```
    GET /api/v1/data/stocks?market=主板&industry=银行&limit=50
    ```
    """
    try:
        loader = StockDataLoader()
        stocks = loader.get_stock_list(
            market=market,
            industry=industry,
            is_active=is_active,
            limit=limit,
            offset=offset
        )
        
        stock_infos = [
            StockInfo(symbol=stock['symbol'], name=stock.get('name'))
            for stock in stocks
        ]
        
        return StockListResponse(
            stocks=stock_infos,
            count=len(stock_infos)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取股票列表失败: {str(e)}")


@router.get(
    "/stocks/{symbol}/kline",
    response_model=KlineDataResponse,
    summary="获取股票K线数据",
    description="获取指定股票的历史K线数据，包括开盘价、收盘价、最高价、最低价、成交量等",
    responses={
        200: {"description": "成功返回K线数据"},
        404: {"description": "未找到指定股票的数据"},
        500: {"description": "服务器内部错误"}
    }
)
async def get_kline_data(
    symbol: str,
    start_date: Optional[str] = Query(None, description="开始日期 (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="结束日期 (YYYY-MM-DD)"),
    limit: int = Query(100, ge=1, le=1000, description="返回数量限制，最大1000"),
    db: Session = Depends(get_db)
):
    """
    获取股票K线数据
    
    获取指定股票的历史K线数据，支持日期范围筛选：
    
    - **symbol**: 股票代码（如：000001）
    - **start_date**: 开始日期（格式：YYYY-MM-DD）
    - **end_date**: 结束日期（格式：YYYY-MM-DD）
    - **limit**: 返回数量限制（1-1000）
    
    **返回数据包括：**
    - 开盘价、收盘价、最高价、最低价
    - 成交量、成交额
    - 涨跌幅、换手率
    - 市盈率、市净率
    
    **示例请求：**
    ```
    GET /api/v1/data/stocks/000001/kline?start_date=2024-01-01&end_date=2024-12-31
    ```
    """
    try:
        loader = StockDataLoader()
        df = loader.load_kline_data(symbol, start_date, end_date, limit=limit)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"未找到股票 {symbol} 的数据")
        
        # 转换为字典列表
        data = df.to_dict('records')
        
        # 转换日期格式
        for item in data:
            if 'trade_date' in item:
                item['trade_date'] = item['trade_date'].strftime('%Y-%m-%d')
        
        return KlineDataResponse(
            symbol=symbol,
            data=data,
            count=len(data)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取K线数据失败: {str(e)}")


@router.get(
    "/stocks/{symbol}/info",
    summary="获取股票基本信息",
    description="获取指定股票的基本信息，包括名称、行业、市值等",
    responses={
        200: {"description": "成功返回股票信息"},
        404: {"description": "未找到指定股票"},
        500: {"description": "服务器内部错误"}
    }
)
async def get_stock_info(
    symbol: str,
    db: Session = Depends(get_db)
):
    """
    获取股票基本信息
    
    获取指定股票的基本信息：
    
    - **symbol**: 股票代码（如：000001）
    
    **返回信息包括：**
    - 股票名称、代码
    - 所属行业、市场
    - 上市日期
    - 总股本、流通股本
    - 总市值、流通市值
    
    **示例请求：**
    ```
    GET /api/v1/data/stocks/000001/info
    ```
    """
    try:
        loader = StockDataLoader()
        info = loader.load_stock_info(symbol)
        
        if not info:
            raise HTTPException(status_code=404, detail=f"未找到股票 {symbol} 的信息")
        
        return info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取股票信息失败: {str(e)}")
