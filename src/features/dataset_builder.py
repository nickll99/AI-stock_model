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
        
        print(f"原始数据: {len(df_features)} 条记录")
        
        # 添加技术指标
        if include_technical:
            df_features = self.feature_engineer.create_technical_indicators(df_features)
            print(f"添加技术指标后: {len(df_features)} 条记录")
        
        # 添加价格特征
        if include_price:
            df_features = self.feature_engineer.create_price_features(df_features)
            print(f"添加价格特征后: {len(df_features)} 条记录")
        
        # 添加成交量特征
        if include_volume:
            df_features = self.feature_engineer.create_volume_features(df_features)
            print(f"添加成交量特征后: {len(df_features)} 条记录")
        
        # 检查NaN情况
        rows_before = len(df_features)
        nan_counts = df_features.isna().sum()
        
        # 先删除完全是NaN的列
        all_nan_cols = nan_counts[nan_counts == rows_before].index.tolist()
        if len(all_nan_cols) > 0:
            print(f"\n删除完全是NaN的列 ({len(all_nan_cols)}个):")
            for col in all_nan_cols:
                print(f"  - {col}")
            df_features = df_features.drop(columns=all_nan_cols)
            # 重新计算NaN统计
            nan_counts = df_features.isna().sum()
        
        # 显示包含NaN的列
        cols_with_nan = nan_counts[nan_counts > 0]
        if len(cols_with_nan) > 0:
            print(f"\n包含NaN的列 ({len(cols_with_nan)}个):")
            # 只显示前10个
            for col, count in list(cols_with_nan.items())[:10]:
                print(f"  {col}: {count} 个NaN ({count/rows_before*100:.1f}%)")
            if len(cols_with_nan) > 10:
                print(f"  ... 还有 {len(cols_with_nan) - 10} 个列包含NaN")
        
        # 尝试删除NaN行
        df_features_dropna = df_features.dropna()
        rows_after_dropna = len(df_features_dropna)
        
        print(f"\n如果删除所有NaN行: 剩余 {rows_after_dropna} 条记录 (删除了 {rows_before - rows_after_dropna} 条)")
        
        # 如果删除NaN后数据太少或为空，使用填充策略
        if rows_after_dropna < 100:
            print(f"\n⚠️  删除NaN后数据太少，使用填充策略...")
            
            # 策略1：前向填充（适用于技术指标）
            df_features = df_features.fillna(method='ffill')
            
            # 策略2：对于仍然是NaN的（开头的数据），使用后向填充
            df_features = df_features.fillna(method='bfill')
            
            # 策略3：如果还有NaN（极端情况），使用0填充
            df_features = df_features.fillna(0)
            
            rows_after = len(df_features)
            print(f"✓ 使用填充策略后: {rows_after} 条记录")
            
            # 检查是否还有NaN
            remaining_nan = df_features.isna().sum().sum()
            if remaining_nan > 0:
                print(f"⚠️  警告: 仍有 {remaining_nan} 个NaN值")
        else:
            df_features = df_features_dropna
            rows_after = rows_after_dropna
            print(f"✓ 删除NaN后: {rows_after} 条记录")
        
        if len(df_features) == 0:
            raise ValueError(
                f"特征构建后数据为空！原始数据有 {len(df)} 条记录，"
                f"但在计算技术指标后变为0条。\n"
                f"这通常是因为数据质量问题或数据量不足。\n"
                f"建议：\n"
                f"  1. 检查数据库中的数据质量\n"
                f"  2. 增加数据量（至少需要200条以上的历史数据）\n"
                f"  3. 或者减少技术指标的计算周期"
            )
        
        if len(df_features) < 100:
            print(f"\n⚠️  警告: 特征数据量较少({len(df_features)}条)，可能影响模型训练效果")
        
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
        
        # 检查数据是否为空
        if len(df_prep) == 0:
            raise ValueError(f"输入数据为空，无法准备序列")
        
        # 确定特征列（在标准化之前）
        if feature_cols is None:
            # 排除非特征列
            exclude_cols = ['symbol', target_col]
            feature_cols = [col for col in df_prep.columns if col not in exclude_cols]
        
        # 检查是否有足够的数据
        if len(df_prep) < seq_length:
            raise ValueError(f"数据量不足：需要至少 {seq_length} 条记录，但只有 {len(df_prep)} 条")
        
        # 标准化
        if normalize:
            df_prep = self.preprocessor.normalize_features(df_prep, method='standard', fit=True)
        
        # 再次检查标准化后的数据
        if len(df_prep) == 0:
            raise ValueError(f"标准化后数据为空，请检查数据质量")
        
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
