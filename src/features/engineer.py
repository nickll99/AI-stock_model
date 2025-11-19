"""特征工程模块"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class FeatureEngineer:
    """特征工程类"""
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成技术指标特征
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            添加技术指标后的DataFrame
        """
        df_feat = df.copy()
        
        # MA - 移动平均线
        for period in [5, 10, 20, 60]:
            df_feat[f'ma_{period}'] = self.calculate_ma(df['close'].values, period)
        
        # EMA - 指数移动平均
        for period in [12, 26]:
            df_feat[f'ema_{period}'] = self.calculate_ema(df['close'].values, period)
        
        # MACD
        macd_dif, macd_dea, macd_bar = self.calculate_macd(df['close'].values)
        df_feat['macd_dif'] = macd_dif
        df_feat['macd_dea'] = macd_dea
        df_feat['macd_bar'] = macd_bar
        
        # RSI - 相对强弱指标
        for period in [6, 12, 24]:
            df_feat[f'rsi_{period}'] = self.calculate_rsi(df['close'].values, period)
        
        # Bollinger Bands - 布林带
        boll_upper, boll_mid, boll_lower = self.calculate_bollinger_bands(df['close'].values)
        df_feat['boll_upper'] = boll_upper
        df_feat['boll_mid'] = boll_mid
        df_feat['boll_lower'] = boll_lower
        
        # KDJ指标
        k, d, j = self.calculate_kdj(df['high'].values, df['low'].values, df['close'].values)
        df_feat['kdj_k'] = k
        df_feat['kdj_d'] = d
        df_feat['kdj_j'] = j
        
        return df_feat
    
    def calculate_ma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        计算移动平均线
        
        Args:
            prices: 价格数组
            period: 周期
            
        Returns:
            MA值数组
        """
        ma = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            ma[i] = np.mean(prices[i - period + 1:i + 1])
        return ma
    
    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        计算指数移动平均
        
        Args:
            prices: 价格数组
            period: 周期
            
        Returns:
            EMA值数组
        """
        ema = np.full(len(prices), np.nan)
        multiplier = 2 / (period + 1)
        
        # 第一个EMA值使用SMA
        ema[period - 1] = np.mean(prices[:period])
        
        # 后续EMA值
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
        ema_fast = self.calculate_ema(prices, fast_period)
        ema_slow = self.calculate_ema(prices, slow_period)
        
        # DIF = EMA(fast) - EMA(slow)
        dif = ema_fast - ema_slow
        
        # DEA = EMA(DIF, signal_period)
        dea = self.calculate_ema(dif[~np.isnan(dif)], signal_period)
        dea_full = np.full(len(prices), np.nan)
        dea_full[slow_period - 1:slow_period - 1 + len(dea)] = dea
        
        # MACD柱 = (DIF - DEA) * 2
        macd_bar = (dif - dea_full) * 2
        
        return dif, dea_full, macd_bar
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        计算RSI指标
        
        Args:
            prices: 价格数组
            period: 周期
            
        Returns:
            RSI值数组
        """
        rsi = np.full(len(prices), np.nan)
        
        # 计算价格变化
        deltas = np.diff(prices)
        
        # 分离上涨和下跌
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # 计算平均涨跌幅
        avg_gain = np.full(len(prices), np.nan)
        avg_loss = np.full(len(prices), np.nan)
        
        # 初始平均值
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        
        # 后续使用平滑平均
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
        
        # 计算RSI
        for i in range(period, len(prices)):
            if avg_loss[i] == 0:
                rsi[i] = 100
            else:
                rs = avg_gain[i] / avg_loss[i]
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
        mid = self.calculate_ma(prices, period)
        
        std = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            std[i] = np.std(prices[i - period + 1:i + 1])
        
        upper = mid + num_std * std
        lower = mid - num_std * std
        
        return upper, mid, lower
    
    def calculate_kdj(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray, 
        period: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算KDJ指标
        
        Args:
            high: 最高价数组
            low: 最低价数组
            close: 收盘价数组
            period: 周期
            
        Returns:
            (K值, D值, J值) 元组
        """
        k = np.full(len(close), 50.0)  # K初始值为50
        d = np.full(len(close), 50.0)  # D初始值为50
        j = np.full(len(close), np.nan)
        
        for i in range(period - 1, len(close)):
            # 计算周期内的最高价和最低价
            highest = np.max(high[i - period + 1:i + 1])
            lowest = np.min(low[i - period + 1:i + 1])
            
            # 计算RSV
            if highest != lowest:
                rsv = (close[i] - lowest) / (highest - lowest) * 100
            else:
                rsv = 50
            
            # 计算K值 (当日K值 = 2/3 × 前一日K值 + 1/3 × 当日RSV)
            k[i] = (2 / 3) * k[i - 1] + (1 / 3) * rsv
            
            # 计算D值 (当日D值 = 2/3 × 前一日D值 + 1/3 × 当日K值)
            d[i] = (2 / 3) * d[i - 1] + (1 / 3) * k[i]
            
            # 计算J值 (J = 3K - 2D)
            j[i] = 3 * k[i] - 2 * d[i]
        
        return k, d, j
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成价格相关特征
        
        Args:
            df: 数据DataFrame
            
        Returns:
            添加价格特征后的DataFrame
        """
        df_feat = df.copy()
        
        # 涨跌幅（如果数据库中没有）
        if 'pct_chg' not in df_feat.columns and 'close' in df_feat.columns:
            df_feat['pct_chg'] = df_feat['close'].pct_change() * 100
        
        # 振幅
        if all(col in df_feat.columns for col in ['high', 'low', 'pre_close']):
            df_feat['amplitude'] = (df_feat['high'] - df_feat['low']) / df_feat['pre_close'] * 100
        
        # 价格变化率
        for period in [1, 3, 5, 10]:
            df_feat[f'price_change_{period}d'] = df_feat['close'].pct_change(period) * 100
        
        # 价格位置（当前价格在最高最低价之间的位置）
        if all(col in df_feat.columns for col in ['close', 'high', 'low']):
            df_feat['price_position'] = (df_feat['close'] - df_feat['low']) / (df_feat['high'] - df_feat['low'])
            df_feat['price_position'] = df_feat['price_position'].fillna(0.5)
        
        # 上下影线比例
        if all(col in df_feat.columns for col in ['open', 'close', 'high', 'low']):
            body = abs(df_feat['close'] - df_feat['open'])
            upper_shadow = df_feat['high'] - df_feat[['open', 'close']].max(axis=1)
            lower_shadow = df_feat[['open', 'close']].min(axis=1) - df_feat['low']
            
            df_feat['upper_shadow_ratio'] = upper_shadow / (body + 1e-10)
            df_feat['lower_shadow_ratio'] = lower_shadow / (body + 1e-10)
        
        return df_feat
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成成交量特征
        
        Args:
            df: 数据DataFrame
            
        Returns:
            添加成交量特征后的DataFrame
        """
        df_feat = df.copy()
        
        if 'vol' not in df_feat.columns:
            return df_feat
        
        # 成交量变化率
        for period in [1, 3, 5, 10]:
            df_feat[f'vol_change_{period}d'] = df_feat['vol'].pct_change(period) * 100
        
        # 量比（当日成交量 / 过去N日平均成交量）
        for period in [5, 10, 20]:
            vol_ma = df_feat['vol'].rolling(window=period).mean()
            df_feat[f'vol_ratio_{period}d'] = df_feat['vol'] / vol_ma
        
        # 成交额相关特征
        if 'amount' in df_feat.columns:
            # 成交额变化率
            df_feat['amount_change_1d'] = df_feat['amount'].pct_change() * 100
            
            # 平均成交价
            df_feat['avg_price'] = df_feat['amount'] * 1000 / (df_feat['vol'] * 100 + 1e-10)
        
        # 换手率相关（如果有）
        if 'turnover_rate' in df_feat.columns:
            # 换手率移动平均
            df_feat['turnover_ma_5'] = df_feat['turnover_rate'].rolling(window=5).mean()
            df_feat['turnover_ma_20'] = df_feat['turnover_rate'].rolling(window=20).mean()
        
        # 量价关系
        if all(col in df_feat.columns for col in ['vol', 'close']):
            # 价涨量增/价跌量减为正向信号
            price_change = df_feat['close'].pct_change()
            vol_change = df_feat['vol'].pct_change()
            df_feat['price_vol_correlation'] = np.sign(price_change) * np.sign(vol_change)
        
        return df_feat
