"""技术指标计算器"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import date

from src.data.loader import StockDataLoader
from src.database.repositories import TechnicalIndicatorRepository
from sqlalchemy.orm import Session


class TechnicalIndicatorCalculator:
    """技术指标计算器"""
    
    def __init__(self, db_session: Optional[Session] = None):
        """
        初始化技术指标计算器
        
        Args:
            db_session: 数据库会话（可选）
        """
        self.data_loader = StockDataLoader()
        self.db_session = db_session
        if db_session:
            self.repository = TechnicalIndicatorRepository(db_session)
    
    def calculate_ma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        计算移动平均线 (MA)
        
        Args:
            prices: 价格数组
            period: 周期
            
        Returns:
            MA值数组
        """
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        ma = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            ma[i] = np.mean(prices[i - period + 1:i + 1])
        
        return ma
    
    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        计算指数移动平均线 (EMA)
        
        Args:
            prices: 价格数组
            period: 周期
            
        Returns:
            EMA值数组
        """
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        ema = np.full(len(prices), np.nan)
        
        # 第一个EMA值使用SMA
        ema[period - 1] = np.mean(prices[:period])
        
        # 计算平滑系数
        multiplier = 2 / (period + 1)
        
        # 计算后续EMA值
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]
        
        return ema
    
    def calculate_macd(
        self,
        prices: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算MACD指标
        
        Args:
            prices: 价格数组
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            
        Returns:
            (DIF, DEA, MACD柱) 元组
        """
        # 计算快线和慢线EMA
        ema_fast = self.calculate_ema(prices, fast_period)
        ema_slow = self.calculate_ema(prices, slow_period)
        
        # DIF = 快线 - 慢线
        dif = ema_fast - ema_slow
        
        # DEA = DIF的EMA
        dea = self.calculate_ema(dif[~np.isnan(dif)], signal_period)
        
        # 对齐数组长度
        dea_aligned = np.full(len(prices), np.nan)
        valid_indices = np.where(~np.isnan(dif))[0]
        if len(valid_indices) >= signal_period:
            dea_aligned[valid_indices[signal_period - 1]:] = dea[signal_period - 1:]
        
        # MACD柱 = (DIF - DEA) * 2
        macd_bar = (dif - dea_aligned) * 2
        
        return dif, dea_aligned, macd_bar
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        计算相对强弱指标 (RSI)
        
        Args:
            prices: 价格数组
            period: 周期
            
        Returns:
            RSI值数组
        """
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        
        # 计算价格变化
        deltas = np.diff(prices)
        
        # 分离上涨和下跌
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # 初始化RSI数组
        rsi = np.full(len(prices), np.nan)
        
        # 计算第一个平均值
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            rsi[period] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[period] = 100 - (100 / (1 + rs))
        
        # 计算后续RSI值
        for i in range(period + 1, len(prices)):
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
            
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(
        self,
        prices: np.ndarray,
        period: int = 20,
        num_std: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算布林带
        
        Args:
            prices: 价格数组
            period: 周期
            num_std: 标准差倍数
            
        Returns:
            (上轨, 中轨, 下轨) 元组
        """
        if len(prices) < period:
            nan_array = np.full(len(prices), np.nan)
            return nan_array, nan_array, nan_array
        
        # 中轨 = MA
        middle_band = self.calculate_ma(prices, period)
        
        # 计算标准差
        std = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            std[i] = np.std(prices[i - period + 1:i + 1])
        
        # 上轨和下轨
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return upper_band, middle_band, lower_band
    
    def calculate_kdj(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 9,
        k_period: int = 3,
        d_period: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算KDJ指标
        
        Args:
            high: 最高价数组
            low: 最低价数组
            close: 收盘价数组
            period: RSV周期
            k_period: K值平滑周期
            d_period: D值平滑周期
            
        Returns:
            (K, D, J) 元组
        """
        if len(close) < period:
            nan_array = np.full(len(close), np.nan)
            return nan_array, nan_array, nan_array
        
        # 计算RSV
        rsv = np.full(len(close), np.nan)
        for i in range(period - 1, len(close)):
            highest = np.max(high[i - period + 1:i + 1])
            lowest = np.min(low[i - period + 1:i + 1])
            
            if highest == lowest:
                rsv[i] = 50
            else:
                rsv[i] = (close[i] - lowest) / (highest - lowest) * 100
        
        # 初始化K和D
        k = np.full(len(close), np.nan)
        d = np.full(len(close), np.nan)
        
        # 第一个K和D值
        first_valid_idx = period - 1
        k[first_valid_idx] = 50
        d[first_valid_idx] = 50
        
        # 计算K和D
        for i in range(first_valid_idx + 1, len(close)):
            k[i] = (k[i - 1] * (k_period - 1) + rsv[i]) / k_period
            d[i] = (d[i - 1] * (d_period - 1) + k[i]) / d_period
        
        # 计算J
        j = 3 * k - 2 * d
        
        return k, d, j
    
    def calculate_all_indicators(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        计算所有技术指标
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含所有技术指标的DataFrame
        """
        # 加载K线数据
        df = self.data_loader.load_kline_data(symbol, start_date, end_date)
        
        if df.empty:
            return pd.DataFrame()
        
        # 提取价格数据
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # 计算各项指标
        # MA
        df['ma5'] = self.calculate_ma(close, 5)
        df['ma10'] = self.calculate_ma(close, 10)
        df['ma20'] = self.calculate_ma(close, 20)
        df['ma60'] = self.calculate_ma(close, 60)
        
        # EMA
        df['ema12'] = self.calculate_ema(close, 12)
        df['ema26'] = self.calculate_ema(close, 26)
        
        # MACD
        dif, dea, macd_bar = self.calculate_macd(close)
        df['macd_dif'] = dif
        df['macd_dea'] = dea
        df['macd_bar'] = macd_bar
        
        # RSI
        df['rsi6'] = self.calculate_rsi(close, 6)
        df['rsi12'] = self.calculate_rsi(close, 12)
        df['rsi24'] = self.calculate_rsi(close, 24)
        
        # 布林带
        upper, middle, lower = self.calculate_bollinger_bands(close, 20)
        df['boll_upper'] = upper
        df['boll_mid'] = middle
        df['boll_lower'] = lower
        
        # KDJ
        k, d, j = self.calculate_kdj(high, low, close)
        df['kdj_k'] = k
        df['kdj_d'] = d
        df['kdj_j'] = j
        
        return df
    
    def calculate_and_save(
        self,
        symbol: str,
        trade_date: date,
        lookback_days: int = 100
    ) -> Dict:
        """
        计算并保存技术指标到数据库
        
        Args:
            symbol: 股票代码
            trade_date: 交易日期
            lookback_days: 回溯天数（用于计算指标）
            
        Returns:
            技术指标字典
        """
        if not self.db_session:
            raise ValueError("需要数据库会话才能保存指标")
        
        # 计算需要的日期范围
        from datetime import timedelta
        start_date = (trade_date - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        end_date = trade_date.strftime('%Y-%m-%d')
        
        # 计算所有指标
        df = self.calculate_all_indicators(symbol, start_date, end_date)
        
        if df.empty:
            raise ValueError(f"无法获取 {symbol} 的数据")
        
        # 获取指定日期的指标
        target_row = df[df['trade_date'] == trade_date]
        
        if target_row.empty:
            raise ValueError(f"未找到 {symbol} 在 {trade_date} 的数据")
        
        row = target_row.iloc[0]
        
        # 构建指标字典
        indicators = {
            'ma5': float(row['ma5']) if not pd.isna(row['ma5']) else None,
            'ma10': float(row['ma10']) if not pd.isna(row['ma10']) else None,
            'ma20': float(row['ma20']) if not pd.isna(row['ma20']) else None,
            'ma60': float(row['ma60']) if not pd.isna(row['ma60']) else None,
            'ema12': float(row['ema12']) if not pd.isna(row['ema12']) else None,
            'ema26': float(row['ema26']) if not pd.isna(row['ema26']) else None,
            'macd_dif': float(row['macd_dif']) if not pd.isna(row['macd_dif']) else None,
            'macd_dea': float(row['macd_dea']) if not pd.isna(row['macd_dea']) else None,
            'macd_bar': float(row['macd_bar']) if not pd.isna(row['macd_bar']) else None,
            'rsi6': float(row['rsi6']) if not pd.isna(row['rsi6']) else None,
            'rsi12': float(row['rsi12']) if not pd.isna(row['rsi12']) else None,
            'rsi24': float(row['rsi24']) if not pd.isna(row['rsi24']) else None,
            'boll_upper': float(row['boll_upper']) if not pd.isna(row['boll_upper']) else None,
            'boll_mid': float(row['boll_mid']) if not pd.isna(row['boll_mid']) else None,
            'boll_lower': float(row['boll_lower']) if not pd.isna(row['boll_lower']) else None,
            'kdj_k': float(row['kdj_k']) if not pd.isna(row['kdj_k']) else None,
            'kdj_d': float(row['kdj_d']) if not pd.isna(row['kdj_d']) else None,
            'kdj_j': float(row['kdj_j']) if not pd.isna(row['kdj_j']) else None,
        }
        
        # 保存到数据库
        self.repository.create_or_update(symbol, trade_date, indicators)
        self.db_session.commit()
        
        return indicators
    
    def get_latest_indicators(self, symbol: str) -> Optional[Dict]:
        """
        获取最新的技术指标
        
        Args:
            symbol: 股票代码
            
        Returns:
            技术指标字典
        """
        if not self.db_session:
            raise ValueError("需要数据库会话才能查询指标")
        
        # 获取最新交易日期
        latest_date = self.data_loader.get_latest_trade_date(symbol)
        
        if not latest_date:
            return None
        
        from datetime import datetime
        trade_date = datetime.strptime(latest_date, '%Y-%m-%d').date()
        
        # 从数据库查询
        indicator = self.repository.get_by_date(symbol, trade_date)
        
        if indicator:
            return {
                'symbol': indicator.symbol,
                'trade_date': indicator.trade_date.strftime('%Y-%m-%d'),
                'ma5': indicator.ma5,
                'ma10': indicator.ma10,
                'ma20': indicator.ma20,
                'ma60': indicator.ma60,
                'ema12': indicator.ema12,
                'ema26': indicator.ema26,
                'macd_dif': indicator.macd_dif,
                'macd_dea': indicator.macd_dea,
                'macd_bar': indicator.macd_bar,
                'rsi6': indicator.rsi6,
                'rsi12': indicator.rsi12,
                'rsi24': indicator.rsi24,
                'boll_upper': indicator.boll_upper,
                'boll_mid': indicator.boll_mid,
                'boll_lower': indicator.boll_lower,
                'kdj_k': indicator.kdj_k,
                'kdj_d': indicator.kdj_d,
                'kdj_j': indicator.kdj_j,
            }
        
        # 如果数据库中没有，则计算并保存
        return self.calculate_and_save(symbol, trade_date)
