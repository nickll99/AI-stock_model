"""特征数据集构建器"""
import pandas as pd
import numpy as np
import pickle
from typing import Tuple, Optional, Dict
from pathlib import Path

from src.features.engineer import FeatureEngineer
from src.data.preprocessor import DataPreprocessor


class FeatureDatasetBuilder:
    """特征数据集构建器"""
    
    def __init__(self, cache_dir: str = "cache/features"):
        """
        初始化构建器
        
        Args:
            cache_dir: 特征缓存目录
        """
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = DataPreprocessor()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def build_feature_matrix(
        self, 
        df: pd.DataFrame,
        include_technical: bool = True,
        include_price: bool = True,
        include_volume: bool = True
    ) -> pd.DataFrame:
        """
        生成特征矩阵
        
        Args:
            df: 原始数据DataFrame
            include_technical: 是否包含技术指标
            include_price: 是否包含价格特征
            include_volume: 是否包含成交量特征
            
        Returns:
            特征矩阵DataFrame
        """
        df_features = df.copy()
        
        # 添加技术指标
        if include_technical:
            df_features = self.feature_engineer.create_technical_indicators(df_features)
        
        # 添加价格特征
        if include_price:
            df_features = self.feature_engineer.create_price_features(df_features)
        
        # 添加成交量特征
        if include_volume:
            df_features = self.feature_engineer.create_volume_features(df_features)
        
        # 删除NaN行（由于计算指标产生的）
        df_features = df_features.dropna()
        
        return df_features
    
    def split_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        划分训练集、验证集、测试集
        
        Args:
            df: 特征DataFrame
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            (train_df, val_df, test_df) 元组
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
        
        n_samples = len(df)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        return train_df, val_df, test_df
    
    def prepare_sequences(
        self,
        df: pd.DataFrame,
        seq_length: int = 60,
        target_col: str = 'close',
        feature_cols: Optional[list] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        准备时间序列数据
        
        Args:
            df: 特征DataFrame
            seq_length: 序列长度
            target_col: 目标列
            feature_cols: 特征列列表
            normalize: 是否标准化
            
        Returns:
            (X, y, feature_names) 元组
        """
        df_prep = df.copy()
        
        # 标准化
        if normalize:
            df_prep = self.preprocessor.normalize_features(df_prep, method='standard', fit=True)
        
        # 确定特征列
        if feature_cols is None:
            # 排除非特征列
            exclude_cols = ['symbol', target_col]
            feature_cols = [col for col in df_prep.columns if col not in exclude_cols]
        
        # 创建序列
        X, y = self.preprocessor.create_sequences(
            df_prep,
            seq_length=seq_length,
            target_col=target_col,
            feature_cols=feature_cols
        )
        
        return X, y, feature_cols
    
    def build_complete_dataset(
        self,
        df: pd.DataFrame,
        seq_length: int = 60,
        target_col: str = 'close',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        normalize: bool = True,
        use_cache: bool = True,
        cache_key: Optional[str] = None
    ) -> Dict:
        """
        构建完整的训练数据集
        
        Args:
            df: 原始数据DataFrame
            seq_length: 序列长度
            target_col: 目标列
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            normalize: 是否标准化
            use_cache: 是否使用缓存
            cache_key: 缓存键（通常使用股票代码）
            
        Returns:
            包含训练、验证、测试数据的字典
        """
        # 检查缓存
        if use_cache and cache_key:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # 1. 构建特征矩阵
        df_features = self.build_feature_matrix(df)
        
        # 2. 划分数据集
        train_df, val_df, test_df = self.split_dataset(
            df_features, train_ratio, val_ratio, test_ratio
        )
        
        # 3. 准备序列数据（训练集）
        X_train, y_train, feature_names = self.prepare_sequences(
            train_df, seq_length, target_col, normalize=normalize
        )
        
        # 4. 准备验证集和测试集（使用训练集的scaler）
        X_val, y_val, _ = self.prepare_sequences(
            val_df, seq_length, target_col, feature_names, normalize=False
        )
        
        X_test, y_test, _ = self.prepare_sequences(
            test_df, seq_length, target_col, feature_names, normalize=False
        )
        
        # 5. 构建返回数据
        dataset = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': feature_names,
            'seq_length': seq_length,
            'target_col': target_col,
            'scaler': self.preprocessor.scaler,
            'train_dates': train_df.index[-len(y_train):].tolist(),
            'val_dates': val_df.index[-len(y_val):].tolist(),
            'test_dates': test_df.index[-len(y_test):].tolist()
        }
        
        # 保存到缓存
        if use_cache and cache_key:
            self._save_to_cache(cache_key, dataset)
        
        return dataset
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}_features.pkl"
    
    def _save_to_cache(self, cache_key: str, data: Dict) -> None:
        """保存数据到缓存"""
        cache_path = self._get_cache_path(cache_key)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """从缓存加载数据"""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"加载缓存失败: {e}")
                return None
        return None
    
    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        """
        清除缓存
        
        Args:
            cache_key: 如果指定，只清除该键的缓存；否则清除所有缓存
        """
        if cache_key:
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
        else:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
    
    def get_feature_importance_names(self, df: pd.DataFrame) -> list:
        """
        获取所有特征名称（用于特征重要性分析）
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            特征名称列表
        """
        df_features = self.build_feature_matrix(df)
        exclude_cols = ['symbol', 'close']
        return [col for col in df_features.columns if col not in exclude_cols]
