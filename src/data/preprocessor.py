"""数据预处理器"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        self.scaler = None
        self.feature_columns = None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗（处理缺失值、异常值）
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            清洗后的DataFrame
        """
        df_clean = df.copy()
        
        # 1. 处理缺失值
        # 价格字段使用前向填充
        price_cols = ['open', 'high', 'low', 'close', 'pre_close']
        for col in price_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(method='ffill')
        
        # 成交量和成交额缺失填充为0
        volume_cols = ['vol', 'amount']
        for col in volume_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0)
        
        # 其他数值字段使用中位数填充
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in price_cols + volume_cols:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
        
        # 2. 处理异常值（使用IQR方法）
        for col in ['open', 'high', 'low', 'close']:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # 将异常值替换为边界值
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        # 3. 确保价格一致性
        if all(col in df_clean.columns for col in ['open', 'high', 'low', 'close']):
            # 确保 high >= max(open, close) 和 low <= min(open, close)
            df_clean['high'] = df_clean[['high', 'open', 'close']].max(axis=1)
            df_clean['low'] = df_clean[['low', 'open', 'close']].min(axis=1)
        
        return df_clean
    
    def normalize_features(
        self, 
        df: pd.DataFrame, 
        method: str = 'standard',
        fit: bool = True
    ) -> pd.DataFrame:
        """
        特征标准化
        
        Args:
            df: 数据DataFrame
            method: 标准化方法 ('standard' 或 'minmax')
            fit: 是否拟合scaler（训练时为True，预测时为False）
            
        Returns:
            标准化后的DataFrame
        """
        df_norm = df.copy()
        
        # 选择需要标准化的数值列
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除不需要标准化的列
        exclude_cols = ['symbol', 'limit_status']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not numeric_cols:
            return df_norm
        
        # 初始化或使用已有的scaler
        if fit or self.scaler is None:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            df_norm[numeric_cols] = self.scaler.fit_transform(df_norm[numeric_cols])
            self.feature_columns = numeric_cols
        else:
            df_norm[numeric_cols] = self.scaler.transform(df_norm[numeric_cols])
        
        return df_norm
    
    def create_sequences(
        self, 
        df: pd.DataFrame, 
        seq_length: int = 60,
        target_col: str = 'close',
        feature_cols: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列样本（滑动窗口）
        
        Args:
            df: 数据DataFrame
            seq_length: 序列长度（时间窗口大小）
            target_col: 目标列名
            feature_cols: 特征列名列表，如果为None则使用所有数值列
            
        Returns:
            (X, y) 元组，X为特征序列，y为目标值
        """
        if len(df) < seq_length + 1:
            raise ValueError(f"数据长度 {len(df)} 小于所需的序列长度 {seq_length + 1}")
        
        # 确定特征列
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in feature_cols:
                feature_cols.remove(target_col)
        
        # 提取特征和目标
        features = df[feature_cols].values
        targets = df[target_col].values
        
        X, y = [], []
        
        # 滑动窗口生成序列
        for i in range(len(df) - seq_length):
            X.append(features[i:i + seq_length])
            y.append(targets[i + seq_length])
        
        return np.array(X), np.array(y)
    
    def split_train_val_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        划分训练集、验证集、测试集
        
        Args:
            X: 特征数据
            y: 目标数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
        
        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def add_lag_features(
        self, 
        df: pd.DataFrame, 
        columns: list, 
        lags: list = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        添加滞后特征
        
        Args:
            df: 数据DataFrame
            columns: 需要添加滞后特征的列
            lags: 滞后期数列表
            
        Returns:
            添加滞后特征后的DataFrame
        """
        df_lag = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for lag in lags:
                df_lag[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # 删除因滞后产生的NaN行
        df_lag = df_lag.dropna()
        
        return df_lag
    
    def add_rolling_features(
        self, 
        df: pd.DataFrame, 
        columns: list, 
        windows: list = [5, 10, 20, 60]
    ) -> pd.DataFrame:
        """
        添加滚动统计特征
        
        Args:
            df: 数据DataFrame
            columns: 需要添加滚动特征的列
            windows: 窗口大小列表
            
        Returns:
            添加滚动特征后的DataFrame
        """
        df_roll = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for window in windows:
                # 滚动均值
                df_roll[f'{col}_ma_{window}'] = df[col].rolling(window=window).mean()
                # 滚动标准差
                df_roll[f'{col}_std_{window}'] = df[col].rolling(window=window).std()
        
        # 删除因滚动产生的NaN行
        df_roll = df_roll.dropna()
        
        return df_roll
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        反标准化
        
        Args:
            data: 标准化后的数据
            
        Returns:
            原始尺度的数据
        """
        if self.scaler is None:
            raise ValueError("Scaler未初始化，请先调用normalize_features")
        
        return self.scaler.inverse_transform(data)
