"""Redis缓存客户端"""
import json
import redis
from typing import Optional, Any, Dict
from datetime import timedelta
import logging

from src.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis缓存客户端"""
    
    def __init__(self):
        """初始化Redis客户端"""
        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        
        # 测试连接
        try:
            self.client.ping()
            logger.info("Redis连接成功")
        except redis.ConnectionError as e:
            logger.error(f"Redis连接失败: {e}")
            raise
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值（自动反序列化JSON）
        """
        try:
            value = self.client.get(key)
            if value is None:
                return None
            
            # 尝试解析JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        except Exception as e:
            logger.error(f"获取缓存失败 key={key}: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值（自动序列化为JSON）
            ttl: 过期时间（秒）
            
        Returns:
            是否成功
        """
        try:
            # 序列化值
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            elif not isinstance(value, str):
                value = str(value)
            
            if ttl:
                return self.client.setex(key, ttl, value)
            else:
                return self.client.set(key, value)
        except Exception as e:
            logger.error(f"设置缓存失败 key={key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            是否成功
        """
        try:
            return self.client.delete(key) > 0
        except Exception as e:
            logger.error(f"删除缓存失败 key={key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        检查键是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            是否存在
        """
        try:
            return self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"检查缓存存在失败 key={key}: {e}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """
        设置过期时间
        
        Args:
            key: 缓存键
            ttl: 过期时间（秒）
            
        Returns:
            是否成功
        """
        try:
            return self.client.expire(key, ttl)
        except Exception as e:
            logger.error(f"设置过期时间失败 key={key}: {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """
        获取剩余过期时间
        
        Args:
            key: 缓存键
            
        Returns:
            剩余秒数（-1表示永不过期，-2表示不存在）
        """
        try:
            return self.client.ttl(key)
        except Exception as e:
            logger.error(f"获取TTL失败 key={key}: {e}")
            return -2
    
    def keys(self, pattern: str) -> list:
        """
        查找匹配的键
        
        Args:
            pattern: 匹配模式
            
        Returns:
            键列表
        """
        try:
            return self.client.keys(pattern)
        except Exception as e:
            logger.error(f"查找键失败 pattern={pattern}: {e}")
            return []
    
    def delete_pattern(self, pattern: str) -> int:
        """
        删除匹配模式的所有键
        
        Args:
            pattern: 匹配模式
            
        Returns:
            删除的键数量
        """
        try:
            keys = self.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"批量删除失败 pattern={pattern}: {e}")
            return 0
    
    def flush_db(self) -> bool:
        """
        清空当前数据库
        
        Returns:
            是否成功
        """
        try:
            return self.client.flushdb()
        except Exception as e:
            logger.error(f"清空数据库失败: {e}")
            return False


class CacheManager:
    """缓存管理器 - 提供业务级别的缓存操作"""
    
    # 缓存键前缀
    PREFIX_PREDICTION = "prediction"
    PREFIX_INDICATOR = "indicators"
    PREFIX_STOCK_LIST = "stocks"
    PREFIX_MODEL = "model"
    
    # 默认TTL（秒）
    TTL_PREDICTION = 3600  # 1小时
    TTL_INDICATOR = 1800  # 30分钟
    TTL_STOCK_LIST = 86400  # 24小时
    TTL_MODEL = 3600  # 1小时
    
    def __init__(self):
        """初始化缓存管理器"""
        self.redis = RedisClient()
    
    def _make_key(self, prefix: str, *parts: str) -> str:
        """
        生成缓存键
        
        Args:
            prefix: 前缀
            parts: 键的各个部分
            
        Returns:
            完整的缓存键
        """
        return f"{prefix}:{':'.join(parts)}"
    
    # 预测结果缓存
    
    def get_prediction(self, symbol: str, model_version: str) -> Optional[Dict]:
        """获取预测结果缓存"""
        key = self._make_key(self.PREFIX_PREDICTION, symbol, model_version)
        return self.redis.get(key)
    
    def set_prediction(
        self,
        symbol: str,
        model_version: str,
        prediction: Dict,
        ttl: Optional[int] = None
    ) -> bool:
        """设置预测结果缓存"""
        key = self._make_key(self.PREFIX_PREDICTION, symbol, model_version)
        return self.redis.set(key, prediction, ttl or self.TTL_PREDICTION)
    
    def delete_prediction(self, symbol: str, model_version: str) -> bool:
        """删除预测结果缓存"""
        key = self._make_key(self.PREFIX_PREDICTION, symbol, model_version)
        return self.redis.delete(key)
    
    def delete_all_predictions(self, symbol: str) -> int:
        """删除指定股票的所有预测缓存"""
        pattern = self._make_key(self.PREFIX_PREDICTION, symbol, "*")
        return self.redis.delete_pattern(pattern)
    
    # 技术指标缓存
    
    def get_indicators(self, symbol: str, date: str) -> Optional[Dict]:
        """获取技术指标缓存"""
        key = self._make_key(self.PREFIX_INDICATOR, symbol, date)
        return self.redis.get(key)
    
    def set_indicators(
        self,
        symbol: str,
        date: str,
        indicators: Dict,
        ttl: Optional[int] = None
    ) -> bool:
        """设置技术指标缓存"""
        key = self._make_key(self.PREFIX_INDICATOR, symbol, date)
        return self.redis.set(key, indicators, ttl or self.TTL_INDICATOR)
    
    def delete_indicators(self, symbol: str, date: str) -> bool:
        """删除技术指标缓存"""
        key = self._make_key(self.PREFIX_INDICATOR, symbol, date)
        return self.redis.delete(key)
    
    def delete_all_indicators(self, symbol: str) -> int:
        """删除指定股票的所有技术指标缓存"""
        pattern = self._make_key(self.PREFIX_INDICATOR, symbol, "*")
        return self.redis.delete_pattern(pattern)
    
    # 股票列表缓存
    
    def get_stock_list(self, list_type: str = "all") -> Optional[list]:
        """获取股票列表缓存"""
        key = self._make_key(self.PREFIX_STOCK_LIST, list_type)
        return self.redis.get(key)
    
    def set_stock_list(
        self,
        stocks: list,
        list_type: str = "all",
        ttl: Optional[int] = None
    ) -> bool:
        """设置股票列表缓存"""
        key = self._make_key(self.PREFIX_STOCK_LIST, list_type)
        return self.redis.set(key, stocks, ttl or self.TTL_STOCK_LIST)
    
    def delete_stock_list(self, list_type: str = "all") -> bool:
        """删除股票列表缓存"""
        key = self._make_key(self.PREFIX_STOCK_LIST, list_type)
        return self.redis.delete(key)
    
    # 模型元数据缓存
    
    def get_model_metadata(self, model_id: str) -> Optional[Dict]:
        """获取模型元数据缓存"""
        key = self._make_key(self.PREFIX_MODEL, model_id, "metadata")
        return self.redis.get(key)
    
    def set_model_metadata(
        self,
        model_id: str,
        metadata: Dict,
        ttl: Optional[int] = None
    ) -> bool:
        """设置模型元数据缓存"""
        key = self._make_key(self.PREFIX_MODEL, model_id, "metadata")
        return self.redis.set(key, metadata, ttl or self.TTL_MODEL)
    
    def delete_model_metadata(self, model_id: str) -> bool:
        """删除模型元数据缓存"""
        key = self._make_key(self.PREFIX_MODEL, model_id, "metadata")
        return self.redis.delete(key)
    
    # 缓存失效策略
    
    def invalidate_stock_cache(self, symbol: str) -> int:
        """
        使指定股票的所有缓存失效
        
        Args:
            symbol: 股票代码
            
        Returns:
            删除的缓存数量
        """
        count = 0
        count += self.delete_all_predictions(symbol)
        count += self.delete_all_indicators(symbol)
        return count
    
    def invalidate_all_cache(self) -> bool:
        """清空所有缓存"""
        return self.redis.flush_db()


# 全局缓存管理器实例
cache_manager = CacheManager()
