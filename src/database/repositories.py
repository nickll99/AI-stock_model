"""数据库仓库层"""
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from typing import List, Optional, Dict
from datetime import date
import uuid

from src.database.models import AIModel, AIPrediction, TechnicalIndicator


class AIModelRepository:
    """AI模型仓库"""
    
    def __init__(self, db: Session):
        """
        初始化仓库
        
        Args:
            db: 数据库会话
        """
        self.db = db
    
    def create(
        self,
        model_id: Optional[str] = None,
        model_name: str = "",
        model_type: str = "",
        version: str = "",
        symbol: Optional[str] = None,
        training_start_date: Optional[date] = None,
        training_end_date: Optional[date] = None,
        hyperparameters: Optional[Dict] = None,
        performance_metrics: Optional[Dict] = None,
        model_path: Optional[str] = None,
        status: str = 'training'
    ) -> AIModel:
        """
        创建AI模型记录
        
        Args:
            model_name: 模型名称
            model_type: 模型类型
            version: 版本号
            symbol: 股票代码
            training_start_date: 训练开始日期
            training_end_date: 训练结束日期
            hyperparameters: 超参数
            performance_metrics: 性能指标
            model_path: 模型路径
            status: 状态
            
        Returns:
            创建的模型对象
        """
        model = AIModel(
            model_id=model_id or str(uuid.uuid4()),
            model_name=model_name,
            model_type=model_type,
            version=version,
            symbol=symbol,
            training_start_date=training_start_date,
            training_end_date=training_end_date,
            hyperparameters=hyperparameters,
            performance_metrics=performance_metrics,
            model_path=model_path,
            status=status
        )
        
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        
        return model
    
    def get_by_id(self, model_id: str) -> Optional[AIModel]:
        """根据ID获取模型"""
        return self.db.query(AIModel).filter(AIModel.model_id == model_id).first()
    
    def get_by_model_id(self, model_id: str) -> Optional[AIModel]:
        """根据model_id获取模型（别名方法）"""
        return self.get_by_id(model_id)
    
    def get_by_symbol(self, symbol: str, model_type: Optional[str] = None) -> List[AIModel]:
        """根据股票代码获取模型列表"""
        query = self.db.query(AIModel).filter(AIModel.symbol == symbol)
        
        if model_type:
            query = query.filter(AIModel.model_type == model_type)
        
        return query.order_by(desc(AIModel.created_at)).all()
    
    def get_latest_by_symbol(self, symbol: str, model_type: str) -> Optional[AIModel]:
        """获取指定股票的最新模型"""
        return self.db.query(AIModel).filter(
            and_(
                AIModel.symbol == symbol,
                AIModel.model_type == model_type,
                AIModel.status == 'completed'
            )
        ).order_by(desc(AIModel.created_at)).first()
    
    def update_status(self, model_id: str, status: str) -> bool:
        """更新模型状态"""
        model = self.get_by_id(model_id)
        if model:
            model.status = status
            self.db.commit()
            return True
        return False
    
    def update_metrics(self, model_id: str, metrics: Dict) -> bool:
        """更新模型性能指标"""
        model = self.get_by_id(model_id)
        if model:
            model.performance_metrics = metrics
            self.db.commit()
            return True
        return False
    
    def update_model_path(self, model_id: str, model_path: str) -> bool:
        """更新模型路径"""
        model = self.get_by_id(model_id)
        if model:
            model.model_path = model_path
            self.db.commit()
            return True
        return False
    
    def list_all(self, status: Optional[str] = None, limit: int = 100) -> List[AIModel]:
        """列出所有模型"""
        query = self.db.query(AIModel)
        
        if status:
            query = query.filter(AIModel.status == status)
        
        return query.order_by(desc(AIModel.created_at)).limit(limit).all()
    
    def list_models(
        self,
        filters: Optional[Dict] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AIModel]:
        """
        根据过滤条件列出模型
        
        Args:
            filters: 过滤条件字典
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            模型列表
        """
        query = self.db.query(AIModel)
        
        if filters:
            if 'symbol' in filters:
                query = query.filter(AIModel.symbol == filters['symbol'])
            if 'model_type' in filters:
                query = query.filter(AIModel.model_type == filters['model_type'])
            if 'status' in filters:
                query = query.filter(AIModel.status == filters['status'])
        
        return query.order_by(desc(AIModel.created_at)).offset(offset).limit(limit).all()
    
    def update(
        self,
        model_id: str,
        **kwargs
    ) -> Optional[AIModel]:
        """
        更新模型信息
        
        Args:
            model_id: 模型ID
            **kwargs: 要更新的字段
            
        Returns:
            更新后的模型对象
        """
        model = self.get_by_id(model_id)
        if not model:
            return None
        
        for key, value in kwargs.items():
            if hasattr(model, key):
                setattr(model, key, value)
        
        self.db.commit()
        self.db.refresh(model)
        
        return model
    
    def delete(self, model_id: str) -> bool:
        """删除模型"""
        model = self.get_by_id(model_id)
        if model:
            self.db.delete(model)
            self.db.commit()
            return True
        return False


class AIPredictionRepository:
    """AI预测仓库"""
    
    def __init__(self, db: Session):
        """
        初始化仓库
        
        Args:
            db: 数据库会话
        """
        self.db = db
    
    def create(
        self,
        model_id: str,
        symbol: str,
        prediction_date: date,
        target_date: date,
        predicted_close: float,
        confidence_lower: Optional[float] = None,
        confidence_upper: Optional[float] = None
    ) -> AIPrediction:
        """
        创建预测记录
        
        Args:
            model_id: 模型ID
            symbol: 股票代码
            prediction_date: 预测生成日期
            target_date: 预测目标日期
            predicted_close: 预测收盘价
            confidence_lower: 置信区间下限
            confidence_upper: 置信区间上限
            
        Returns:
            创建的预测对象
        """
        prediction = AIPrediction(
            model_id=model_id,
            symbol=symbol,
            prediction_date=prediction_date,
            target_date=target_date,
            predicted_close=predicted_close,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper
        )
        
        self.db.add(prediction)
        self.db.commit()
        self.db.refresh(prediction)
        
        return prediction
    
    def batch_create(self, predictions: List[Dict]) -> int:
        """
        批量创建预测记录
        
        Args:
            predictions: 预测字典列表
            
        Returns:
            创建的记录数
        """
        prediction_objects = [AIPrediction(**pred) for pred in predictions]
        self.db.bulk_save_objects(prediction_objects)
        self.db.commit()
        return len(prediction_objects)
    
    def get_by_symbol_and_date(
        self,
        symbol: str,
        target_date: date,
        model_id: Optional[str] = None
    ) -> List[AIPrediction]:
        """获取指定股票和日期的预测"""
        query = self.db.query(AIPrediction).filter(
            and_(
                AIPrediction.symbol == symbol,
                AIPrediction.target_date == target_date
            )
        )
        
        if model_id:
            query = query.filter(AIPrediction.model_id == model_id)
        
        return query.all()
    
    def get_predictions_range(
        self,
        symbol: str,
        start_date: date,
   
     end_date: date,
        model_id: Optional[str] = None
    ) -> List[AIPrediction]:
        """获取指定日期范围的预测"""
        query = self.db.query(AIPrediction).filter(
            and_(
                AIPrediction.symbol == symbol,
                AIPrediction.target_date >= start_date,
                AIPrediction.target_date <= end_date
            )
        )
        
        if model_id:
            query = query.filter(AIPrediction.model_id == model_id)
        
        return query.order_by(AIPrediction.target_date).all()
    
    def update_actual_price(
        self,
        symbol: str,
        target_date: date,
        actual_close: float
    ) -> int:
        """
        更新实际价格和预测误差
        
        Args:
            symbol: 股票代码
            target_date: 目标日期
            actual_close: 实际收盘价
            
        Returns:
            更新的记录数
        """
        predictions = self.get_by_symbol_and_date(symbol, target_date)
        
        count = 0
        for pred in predictions:
            pred.actual_close = actual_close
            pred.prediction_error = abs(pred.predicted_close - actual_close)
            count += 1
        
        if count > 0:
            self.db.commit()
        
        return count


class TechnicalIndicatorRepository:
    """技术指标仓库"""
    
    def __init__(self, db: Session):
        """
        初始化仓库
        
        Args:
            db: 数据库会话
        """
        self.db = db
    
    def create_or_update(
        self,
        symbol: str,
        trade_date: date,
        indicators: Dict
    ) -> TechnicalIndicator:
        """
        创建或更新技术指标
        
        Args:
            symbol: 股票代码
            trade_date: 交易日期
            indicators: 指标字典
            
        Returns:
            技术指标对象
        """
        # 查找是否已存在
        existing = self.db.query(TechnicalIndicator).filter(
            and_(
                TechnicalIndicator.symbol == symbol,
                TechnicalIndicator.trade_date == trade_date
            )
        ).first()
        
        if existing:
            # 更新
            for key, value in indicators.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            indicator = existing
        else:
            # 创建
            indicator = TechnicalIndicator(
                symbol=symbol,
                trade_date=trade_date,
                **indicators
            )
            self.db.add(indicator)
        
        self.db.commit()
        self.db.refresh(indicator)
        
        return indicator
    
    def get_by_symbol_and_date(
        self,
        symbol: str,
        trade_date: date
    ) -> Optional[TechnicalIndicator]:
        """获取指定股票和日期的技术指标"""
        return self.db.query(TechnicalIndicator).filter(
            and_(
                TechnicalIndicator.symbol == symbol,
                TechnicalIndicator.trade_date == trade_date
            )
        ).first()
    
    def get_range(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> List[TechnicalIndicator]:
        """获取指定日期范围的技术指标"""
        return self.db.query(TechnicalIndicator).filter(
            and_(
                TechnicalIndicator.symbol == symbol,
                TechnicalIndicator.trade_date >= start_date,
                TechnicalIndicator.trade_date <= end_date
            )
        ).order_by(TechnicalIndicator.trade_date).all()
    
    def batch_create_or_update(self, indicators: List[Dict]) -> int:
        """
        批量创建或更新技术指标
        
        Args:
            indicators: 指标字典列表
            
        Returns:
            处理的记录数
        """
        count = 0
        for ind in indicators:
            symbol = ind.pop('symbol')
            trade_date = ind.pop('trade_date')
            self.create_or_update(symbol, trade_date, ind)
            count += 1
        
        return count
