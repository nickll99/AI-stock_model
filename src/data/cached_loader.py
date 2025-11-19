"""
带缓存的数据加载器 - 优化大规模训练性能
"""
import pandas as pd
import pickle
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging

from src.data.loader import StockDataLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ParquetDataLoader:
    """
    Parquet格式缓存的数据加载器
    
    优势：
    - 列式存储，查询速度快
    - 压缩率高，节省磁盘空间
    - 支持增量更新
    """
    
    def __init__(self, cache_dir: str = "data/parquet", cache_ttl: int = 86400):
        """
        初始化Parquet数据加载器
        
        Args:
            cache_dir: 缓存目录
            cache_ttl: 缓存有效期（秒），默认24小时
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl
        self.loader = StockDataLoader()
        
        logger.info(f"Parquet缓存目录: {self.cache_dir}")
    
    def load_kline_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        加载K线数据（优先从缓存）
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            force_refresh: 是否强制刷新缓存
            
        Returns:
            K线数据DataFrame
        """
        cache_file = self.cache_dir / f"{symbol}.parquet"
        
        # 检查缓存
        if use_cache and not force_refresh and cache_file.exists():
            # 检查缓存是否过期
            cache_age = time.time() - cache_file.stat().st_mtime
            
            if cache_age < self.cache_ttl:
                try:
                    # 从缓存加载
                    start_time = time.time()
                    df = pd.read_parquet(cache_file)
                    
                    # 过滤日期范围
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    
                    elapsed_time = time.time() - start_time
                    logger.info(f"从缓存加载 {symbol}: {len(df)} 条记录, 耗时 {elapsed_time:.3f}秒")
                    
                    if not df.empty:
                        return df
                except Exception as e:
                    logger.warning(f"读取缓存失败 {symbol}: {e}, 将从数据库加载")
        
        # 从数据库加载
        start_time = time.time()
        df = self.loader.load_kline_data(symbol, start_date, end_date)
        elapsed_time = time.time() - start_time
        
        logger.info(f"从MySQL加载 {symbol}: {len(df)} 条记录, 耗时 {elapsed_time:.3f}秒")
        
        # 保存缓存
        if use_cache and not df.empty:
            try:
                df.to_parquet(cache_file, compression='snappy')
                logger.info(f"缓存已保存: {cache_file}")
            except Exception as e:
                logger.warning(f"保存缓存失败 {symbol}: {e}")
        
        return df
    
    def load_kline_data_batch(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        批量加载多只股票的K线数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            
        Returns:
            {symbol: DataFrame} 字典
        """
        result = {}
        
        # 分离缓存命中和未命中的股票
        cached_symbols = []
        uncached_symbols = []
        
        for symbol in symbols:
            cache_file = self.cache_dir / f"{symbol}.parquet"
            
            if use_cache and cache_file.exists():
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age < self.cache_ttl:
                    cached_symbols.append(symbol)
                else:
                    uncached_symbols.append(symbol)
            else:
                uncached_symbols.append(symbol)
        
        # 从缓存加载
        if cached_symbols:
            logger.info(f"从缓存加载 {len(cached_symbols)} 只股票")
            for symbol in cached_symbols:
                try:
                    df = self.load_kline_data(symbol, start_date, end_date, use_cache=True)
                    if not df.empty:
                        result[symbol] = df
                except Exception as e:
                    logger.warning(f"缓存加载失败 {symbol}: {e}")
                    uncached_symbols.append(symbol)
        
        # 从数据库批量加载未缓存的股票
        if uncached_symbols:
            logger.info(f"从MySQL批量加载 {len(uncached_symbols)} 只股票")
            
            try:
                # 使用批量加载方法
                batch_data = self.loader.load_kline_data_batch(
                    uncached_symbols,
                    start_date,
                    end_date
                )
                
                # 保存缓存
                for symbol, df in batch_data.items():
                    result[symbol] = df
                    
                    if use_cache and not df.empty:
                        try:
                            cache_file = self.cache_dir / f"{symbol}.parquet"
                            df.to_parquet(cache_file, compression='snappy')
                        except Exception as e:
                            logger.warning(f"保存缓存失败 {symbol}: {e}")
                
            except AttributeError:
                # 如果没有批量加载方法，逐个加载
                logger.warning("批量加载方法不可用，使用逐个加载")
                for symbol in uncached_symbols:
                    try:
                        df = self.load_kline_data(symbol, start_date, end_date, use_cache=use_cache)
                        if not df.empty:
                            result[symbol] = df
                    except Exception as e:
                        logger.error(f"加载失败 {symbol}: {e}")
        
        logger.info(f"批量加载完成: {len(result)}/{len(symbols)} 只股票")
        
        return result
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        清除缓存
        
        Args:
            symbol: 股票代码，如果为None则清除所有缓存
        """
        if symbol:
            cache_file = self.cache_dir / f"{symbol}.parquet"
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"缓存已清除: {symbol}")
        else:
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
            logger.info("所有缓存已清除")
    
    def get_cache_info(self) -> Dict:
        """
        获取缓存信息
        
        Returns:
            缓存统计信息
        """
        cache_files = list(self.cache_dir.glob("*.parquet"))
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "cache_dir": str(self.cache_dir),
            "cached_stocks": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_ttl_hours": self.cache_ttl / 3600
        }


class FeatureCache:
    """
    特征缓存管理器
    
    缓存已计算的特征，避免重复计算技术指标
    """
    
    def __init__(self, cache_dir: str = "data/features", cache_ttl: int = 86400):
        """
        初始化特征缓存管理器
        
        Args:
            cache_dir: 缓存目录
            cache_ttl: 缓存有效期（秒）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl
        
        logger.info(f"特征缓存目录: {self.cache_dir}")
    
    def get_features(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        获取特征（优先从缓存）
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            
        Returns:
            特征DataFrame，如果不存在返回None
        """
        cache_file = self.cache_dir / f"{symbol}_features.parquet"
        
        if use_cache and cache_file.exists():
            # 检查缓存是否过期
            cache_age = time.time() - cache_file.stat().st_mtime
            
            if cache_age < self.cache_ttl:
                try:
                    df = pd.read_parquet(cache_file)
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    
                    if not df.empty:
                        logger.info(f"从缓存加载特征 {symbol}: {len(df)} 条记录")
                        return df
                except Exception as e:
                    logger.warning(f"读取特征缓存失败 {symbol}: {e}")
        
        return None
    
    def save_features(self, symbol: str, df_features: pd.DataFrame):
        """
        保存特征到缓存
        
        Args:
            symbol: 股票代码
            df_features: 特征DataFrame
        """
        if df_features.empty:
            return
        
        try:
            cache_file = self.cache_dir / f"{symbol}_features.parquet"
            df_features.to_parquet(cache_file, compression='snappy')
            logger.info(f"特征缓存已保存: {cache_file}")
        except Exception as e:
            logger.warning(f"保存特征缓存失败 {symbol}: {e}")
    
    def load_features(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        从缓存加载特征
        
        Args:
            symbol: 股票代码
            
        Returns:
            特征DataFrame，如果不存在返回None
        """
        cache_file = self.cache_dir / f"{symbol}_features.parquet"
        
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"从缓存加载特征 {symbol}: {len(df)} 条记录")
                return df
            except Exception as e:
                logger.warning(f"读取特征缓存失败 {symbol}: {e}")
                return None
        
        return None
    
    def clear_cache(self, symbol: Optional[str] = None):
        """清除特征缓存"""
        if symbol:
            cache_file = self.cache_dir / f"{symbol}_features.parquet"
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"特征缓存已清除: {symbol}")
        else:
            for cache_file in self.cache_dir.glob("*_features.parquet"):
                cache_file.unlink()
            logger.info("所有特征缓存已清除")
    
    def get_cache_info(self) -> Dict:
        """获取缓存信息"""
        cache_files = list(self.cache_dir.glob("*_features.parquet"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "cache_dir": str(self.cache_dir),
            "cached_stocks": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_ttl_hours": self.cache_ttl / 3600
        }
