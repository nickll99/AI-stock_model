"""股票数据加载器"""
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from src.database.models import StockBasicInfo, StockKlineData
from src.database.connection import get_db_context


class StockDataLoader:
    """股票数据加载器（从MySQL数据库加载）"""
    
    def __init__(self, db: Optional[Session] = None):
        """
        初始化数据加载器
        
        Args:
            db: 数据库会话，如果为None则使用上下文管理器
        """
        self.db = db
        self._use_context = db is None
    
    def load_kline_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        从stock_kline_data表加载K线数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            包含K线数据的DataFrame
        """
        if self._use_context:
            with get_db_context() as db:
                return self._load_kline_data_impl(db, symbol, start_date, end_date)
        else:
            return self._load_kline_data_impl(self.db, symbol, start_date, end_date)
    
    def _load_kline_data_impl(
        self, 
        db: Session, 
        symbol: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """K线数据加载实现"""
        query = db.query(StockKlineData).filter(
            and_(
                StockKlineData.symbol == symbol,
                StockKlineData.trade_date >= start_date,
                StockKlineData.trade_date <= end_date
            )
        ).order_by(StockKlineData.trade_date)
        
        results = query.all()
        
        if not results:
            return pd.DataFrame()
        
        # 转换为DataFrame
        data = []
        for row in results:
            data.append({
                'symbol': row.symbol,
                'trade_date': row.trade_date,
                'open': float(row.open) if row.open else None,
                'high': float(row.high) if row.high else None,
                'low': float(row.low) if row.low else None,
                'close': float(row.close) if row.close else None,
                'pre_close': float(row.pre_close) if row.pre_close else None,
                'change': float(row.change) if row.change else None,
                'pct_chg': float(row.pct_chg) if row.pct_chg else None,
                'vol': row.vol,
                'amount': float(row.amount) if row.amount else None,
                'turnover_rate': float(row.turnover_rate) if row.turnover_rate else None,
                'pe': float(row.pe) if row.pe else None,
                'pb': float(row.pb) if row.pb else None,
                'total_mv': float(row.total_mv) if row.total_mv else None,
                'circ_mv': float(row.circ_mv) if row.circ_mv else None,
                'limit_status': row.limit_status
            })
        
        df = pd.DataFrame(data)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        
        return df
    
    def load_stock_info(self, symbol: str) -> Dict:
        """
        从stock_basic_info表加载股票基本信息
        
        Args:
            symbol: 股票代码
            
        Returns:
            包含股票信息的字典
        """
        if self._use_context:
            with get_db_context() as db:
                return self._load_stock_info_impl(db, symbol)
        else:
            return self._load_stock_info_impl(self.db, symbol)
    
    def _load_stock_info_impl(self, db: Session, symbol: str) -> Dict:
        """股票信息加载实现"""
        stock = db.query(StockBasicInfo).filter(
            StockBasicInfo.symbol == symbol
        ).first()
        
        if not stock:
            return {}
        
        return {
            'symbol': stock.symbol,
            'name': stock.name,
            'exchange': stock.exchange,
            'market': stock.market,
            'stock_type': stock.stock_type,
            'industry': stock.industry,
            'list_date': stock.list_date,
            'delist_date': stock.delist_date,
            'is_active': stock.is_active,
            'total_share': float(stock.total_share) if stock.total_share else None,
            'float_share': float(stock.float_share) if stock.float_share else None,
            'total_market_value': float(stock.total_market_value) if stock.total_market_value else None,
            'float_market_value': float(stock.float_market_value) if stock.float_market_value else None,
            'province': stock.province,
            'city': stock.city,
            'reg_capital': float(stock.reg_capital) if stock.reg_capital else None,
            'employee_count': stock.employee_count
        }
    
    def get_all_active_stocks(self, stock_type: Optional[str] = None) -> List[str]:
        """
        获取所有活跃股票代码列表（is_active=1）
        
        Args:
            stock_type: 股票类型筛选（如'主板'、'创业板'、'科创板'等），None表示不筛选
        
        Returns:
            股票代码列表
        """
        if self._use_context:
            with get_db_context() as db:
                return self._get_all_active_stocks_impl(db, stock_type)
        else:
            return self._get_all_active_stocks_impl(self.db, stock_type)
    
    def _get_all_active_stocks_impl(self, db: Session, stock_type: Optional[str] = None) -> List[str]:
        """活跃股票列表加载实现"""
        query = db.query(StockBasicInfo.symbol).filter(
            StockBasicInfo.is_active == 1
        )
        
        # 如果指定了股票类型，添加筛选条件
        if stock_type:
            query = query.filter(StockBasicInfo.stock_type == stock_type)
        
        stocks = query.all()
        
        return [stock.symbol for stock in stocks]
    
    def validate_data_completeness(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        min_completeness: float = 0.95
    ) -> bool:
        """
        验证指定时间段数据完整性
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            min_completeness: 最小完整度阈值（默认95%）
            
        Returns:
            数据是否完整
        """
        df = self.load_kline_data(symbol, start_date, end_date)
        
        if df.empty:
            return False
        
        # 计算预期交易日数量（粗略估计：一年约250个交易日）
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        days_diff = (end - start).days
        expected_trading_days = int(days_diff * 250 / 365)
        
        # 实际数据行数
        actual_days = len(df)
        
        # 计算完整度
        completeness = actual_days / expected_trading_days if expected_trading_days > 0 else 0
        
        return completeness >= min_completeness
    
    def get_latest_trade_date(self, symbol: str) -> Optional[str]:
        """
        获取指定股票的最新交易日期
        
        Args:
            symbol: 股票代码
            
        Returns:
            最新交易日期字符串，如果没有数据则返回None
        """
        if self._use_context:
            with get_db_context() as db:
                return self._get_latest_trade_date_impl(db, symbol)
        else:
            return self._get_latest_trade_date_impl(self.db, symbol)
    
    def _get_latest_trade_date_impl(self, db: Session, symbol: str) -> Optional[str]:
        """最新交易日期查询实现"""
        result = db.query(func.max(StockKlineData.trade_date)).filter(
            StockKlineData.symbol == symbol
        ).scalar()
        
        return result.strftime('%Y-%m-%d') if result else None
    
    def get_date_range(self, symbol: str) -> Dict[str, Optional[str]]:
        """
        获取指定股票的数据日期范围
        
        Args:
            symbol: 股票代码
            
        Returns:
            包含start_date和end_date的字典
        """
        if self._use_context:
            with get_db_context() as db:
                return self._get_date_range_impl(db, symbol)
        else:
            return self._get_date_range_impl(self.db, symbol)
    
    def _get_date_range_impl(self, db: Session, symbol: str) -> Dict[str, Optional[str]]:
        """日期范围查询实现"""
        result = db.query(
            func.min(StockKlineData.trade_date).label('start_date'),
            func.max(StockKlineData.trade_date).label('end_date')
        ).filter(
            StockKlineData.symbol == symbol
        ).first()
        
        return {
            'start_date': result.start_date.strftime('%Y-%m-%d') if result.start_date else None,
            'end_date': result.end_date.strftime('%Y-%m-%d') if result.end_date else None
        }
