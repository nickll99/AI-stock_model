"""预测引擎"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats

from src.models.lstm_model import LSTMModel
from src.models.gru_model import GRUModel
from src.models.transformer_model import TransformerModel
from src.data.loader import StockDataLoader
from src.features.dataset_builder import FeatureDatasetBuilder


class PredictionEngine:
    """预测引擎"""
    
    def __init__(
        self,
        model_path: str,
        model_type: str,
        model_config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化预测引擎
        
        Args:
            model_path: 模型文件路径
            model_type: 模型类型 ('lstm', 'gru', 'transformer')
            model_config: 模型配置字典
            device: 设备
        """
        self.model_type = model_type
        self.model_config = model_config
        self.device = device
        
        # 加载模型
        self.model = self._load_model(model_path, model_type, model_config)
        self.model.eval()
        
        # 数据加载器和特征构建器
        self.data_loader = StockDataLoader()
        self.feature_builder = FeatureDatasetBuilder()
    
    def _load_model(
        self,
        model_path: str,
        model_type: str,
        config: Dict
    ) -> nn.Module:
        """加载模型"""
        # 创建模型实例
        if model_type == 'lstm':
            model = LSTMModel(
                input_size=config['input_size'],
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 2),
                output_size=config.get('output_size', 1),
                dropout=config.get('dropout', 0.2),
                bidirectional=config.get('bidirectional', False)
            )
        elif model_type == 'gru':
            model = GRUModel(
                input_size=config['input_size'],
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 2),
                output_size=config.get('output_size', 1),
                dropout=config.get('dropout', 0.2),
                bidirectional=config.get('bidirectional', False)
            )
        elif model_type == 'transformer':
            model = TransformerModel(
                input_size=config['input_size'],
                d_model=config.get('d_model', 128),
                nhead=config.get('nhead', 8),
                num_layers=config.get('num_layers', 3),
                dim_feedforward=config.get('dim_feedforward', 512),
                output_size=config.get('output_size', 1),
                dropout=config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model.to(self.device)
    
    def predict(
        self,
        symbol: str,
        days: int = 5,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        生成未来N天的价格预测
        
        Args:
            symbol: 股票代码
            days: 预测天数
            end_date: 数据截止日期（默认为最新日期）
            
        Returns:
            预测结果字典
        """
        # 获取历史数据
        if end_date is None:
            end_date = self.data_loader.get_latest_trade_date(symbol)
        
        # 计算需要的历史数据长度
        seq_length = self.model_config.get('seq_length', 60)
        lookback_days = seq_length + 100  # 额外的数据用于计算技术指标
        
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        df = self.data_loader.load_kline_data(symbol, start_date, end_date)
        
        if df.empty or len(df) < seq_length:
            raise ValueError(f"数据不足，需要至少 {seq_length} 条记录")
        
        # 构建特征
        df_features = self.feature_builder.build_feature_matrix(df)
        
        # 准备输入数据
        X, _, feature_names = self.feature_builder.prepare_sequences(
            df_features,
            seq_length=seq_length,
            target_col='close',
            normalize=True
        )
        
        # 使用最后一个序列进行预测
        last_sequence = X[-1:]
        
        # 多步预测
        predictions = []
        current_sequence = torch.FloatTensor(last_sequence).to(self.device)
        
        for _ in range(days):
            with torch.no_grad():
                pred = self.model(current_sequence)
                if pred.dim() == 2 and pred.size(1) == 1:
                    pred = pred.squeeze(1)
                
                pred_value = pred.cpu().numpy()[0]
                predictions.append(float(pred_value))
                
                # 更新序列（简化版本，实际应该更新所有特征）
                # 这里只是示例，实际应用中需要更复杂的逻辑
                if len(predictions) < days:
                    # 滚动窗口
                    new_sequence = current_sequence[:, 1:, :].clone()
                    # 这里简化处理，实际需要根据预测值更新所有特征
                    current_sequence = new_sequence
        
        # 反标准化
        if self.feature_builder.preprocessor.scaler is not None:
            # 简化的反标准化（实际需要更精确的处理）
            last_close = float(df['close'].iloc[-1])
            predictions = [last_close * (1 + (p - predictions[0]) * 0.1) for p in predictions]
        
        # 生成预测日期
        last_date = datetime.strptime(end_date, '%Y-%m-%d')
        pred_dates = []
        for i in range(1, days + 1):
            # 简化处理，实际应该跳过周末和节假日
            pred_date = last_date + timedelta(days=i)
            pred_dates.append(pred_date.strftime('%Y-%m-%d'))
        
        # 判断趋势
        trend = self._determine_trend(predictions)
        
        return {
            'symbol': symbol,
            'base_date': end_date,
            'base_price': float(df['close'].iloc[-1]),
            'predictions': [
                {
                    'date': date,
                    'price': price
                }
                for date, price in zip(pred_dates, predictions)
            ],
            'trend': trend,
            'model_type': self.model_type
        }
    
    def batch_predict(
        self,
        symbols: List[str],
        days: int = 5
    ) -> Dict[str, Dict]:
        """
        批量预测多只股票
        
        Args:
            symbols: 股票代码列表
            days: 预测天数
            
        Returns:
            预测结果字典
        """
        results = {}
        
        for symbol in symbols:
            try:
                result = self.predict(symbol, days)
                results[symbol] = result
            except Exception as e:
                print(f"预测 {symbol} 失败: {e}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def predict_with_confidence(
        self,
        symbol: str,
        days: int = 5,
        confidence_level: float = 0.95,
        n_samples: int = 100,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        生成带置信区间的预测
        
        使用蒙特卡洛dropout方法生成预测分布
        
        Args:
            symbol: 股票代码
            days: 预测天数
            confidence_level: 置信水平 (0-1)
            n_samples: 蒙特卡洛采样次数
            end_date: 数据截止日期
            
        Returns:
            包含置信区间的预测结果
        """
        # 获取历史数据
        if end_date is None:
            end_date = self.data_loader.get_latest_trade_date(symbol)
        
        seq_length = self.model_config.get('seq_length', 60)
        lookback_days = seq_length + 100
        
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        df = self.data_loader.load_kline_data(symbol, start_date, end_date)
        
        if df.empty or len(df) < seq_length:
            raise ValueError(f"数据不足，需要至少 {seq_length} 条记录")
        
        # 构建特征
        df_features = self.feature_builder.build_feature_matrix(df)
        X, _, _ = self.feature_builder.prepare_sequences(
            df_features,
            seq_length=seq_length,
            target_col='close',
            normalize=True
        )
        
        # 使用最后一个序列
        last_sequence = torch.FloatTensor(X[-1:]).to(self.device)
        
        # 蒙特卡洛dropout采样
        all_predictions = []
        
        # 启用dropout进行多次预测
        self._enable_dropout()
        
        for _ in range(n_samples):
            predictions = []
            current_sequence = last_sequence.clone()
            
            for _ in range(days):
                with torch.no_grad():
                    pred = self.model(current_sequence)
                    if pred.dim() == 2 and pred.size(1) == 1:
                        pred = pred.squeeze(1)
                    
                    pred_value = pred.cpu().numpy()[0]
                    predictions.append(float(pred_value))
                    
                    # 滚动窗口
                    if len(predictions) < days:
                        new_sequence = current_sequence[:, 1:, :].clone()
                        current_sequence = new_sequence
            
            all_predictions.append(predictions)
        
        # 恢复eval模式
        self.model.eval()
        
        # 转换为numpy数组
        all_predictions = np.array(all_predictions)  # shape: (n_samples, days)
        
        # 计算统计量
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)
        
        # 计算置信区间
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)
        
        lower_bound = mean_predictions - z_score * std_predictions
        upper_bound = mean_predictions + z_score * std_predictions
        
        # 反标准化（简化版本）
        last_close = float(df['close'].iloc[-1])
        if self.feature_builder.preprocessor.scaler is not None:
            mean_predictions = [last_close * (1 + (p - mean_predictions[0]) * 0.1) for p in mean_predictions]
            lower_bound = [last_close * (1 + (p - lower_bound[0]) * 0.1) for p in lower_bound]
            upper_bound = [last_close * (1 + (p - upper_bound[0]) * 0.1) for p in upper_bound]
        
        # 生成预测日期
        last_date = datetime.strptime(end_date, '%Y-%m-%d')
        pred_dates = []
        for i in range(1, days + 1):
            pred_date = last_date + timedelta(days=i)
            pred_dates.append(pred_date.strftime('%Y-%m-%d'))
        
        # 判断趋势
        trend = self._determine_trend(mean_predictions.tolist())
        
        # 计算置信度分数
        confidence_score = self._calculate_confidence_score(std_predictions)
        
        return {
            'symbol': symbol,
            'base_date': end_date,
            'base_price': last_close,
            'predictions': [
                {
                    'date': date,
                    'price': float(price),
                    'confidence_lower': float(lower),
                    'confidence_upper': float(upper),
                    'std': float(std)
                }
                for date, price, lower, upper, std in zip(
                    pred_dates, mean_predictions, lower_bound, upper_bound, std_predictions
                )
            ],
            'trend': trend,
            'confidence_level': confidence_level,
            'confidence_score': confidence_score,
            'model_type': self.model_type,
            'n_samples': n_samples
        }
    
    def _enable_dropout(self):
        """启用模型的dropout层（用于蒙特卡洛dropout）"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def _calculate_confidence_score(self, std_predictions: np.ndarray) -> float:
        """
        计算置信度分数
        
        Args:
            std_predictions: 预测的标准差数组
            
        Returns:
            置信度分数 (0-1)，越高表示预测越可靠
        """
        # 使用标准差的倒数作为置信度
        # 标准差越小，置信度越高
        avg_std = np.mean(std_predictions)
        
        # 归一化到0-1范围
        # 假设标准差在0-1范围内
        confidence = 1.0 / (1.0 + avg_std)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _determine_trend(self, predictions: List[float]) -> str:
        """
        判断趋势方向
        
        Args:
            predictions: 预测价格列表
            
        Returns:
            趋势 ('upward', 'downward', 'stable')
        """
        if len(predictions) < 2:
            return 'stable'
        
        # 计算平均变化率
        changes = [predictions[i] - predictions[i-1] for i in range(1, len(predictions))]
        avg_change = np.mean(changes)
        
        # 计算变化率的标准差
        std_change = np.std(changes)
        
        # 判断趋势
        if abs(avg_change) < std_change * 0.5:
            return 'stable'
        elif avg_change > 0:
            return 'upward'
        else:
            return 'downward'
    
    def _determine_trend_with_confidence(
        self,
        predictions: List[float],
        lower_bound: List[float],
        upper_bound: List[float]
    ) -> Tuple[str, float]:
        """
        基于置信区间判断趋势方向和置信度
        
        Args:
            predictions: 预测价格列表
            lower_bound: 置信区间下界
            upper_bound: 置信区间上界
            
        Returns:
            (趋势, 趋势置信度)
        """
        if len(predictions) < 2:
            return 'stable', 0.5
        
        # 计算首尾价格变化
        start_price = predictions[0]
        end_price = predictions[-1]
        price_change_pct = (end_price - start_price) / start_price
        
        # 检查置信区间是否支持趋势判断
        start_lower = lower_bound[0]
        start_upper = upper_bound[0]
        end_lower = lower_bound[-1]
        end_upper = upper_bound[-1]
        
        # 判断趋势
        if abs(price_change_pct) < 0.02:  # 变化小于2%
            trend = 'stable'
            # 稳定趋势的置信度基于区间宽度
            avg_interval_width = np.mean([u - l for u, l in zip(upper_bound, lower_bound)])
            trend_confidence = 1.0 / (1.0 + avg_interval_width / start_price)
        elif price_change_pct > 0:
            trend = 'upward'
            # 上涨趋势：检查下界是否也上涨
            if end_lower > start_lower:
                trend_confidence = 0.9
            elif end_lower > start_price:
                trend_confidence = 0.7
            else:
                trend_confidence = 0.5
        else:
            trend = 'downward'
            # 下跌趋势：检查上界是否也下跌
            if end_upper < start_upper:
                trend_confidence = 0.9
            elif end_upper < start_price:
                trend_confidence = 0.7
            else:
                trend_confidence = 0.5
        
        return trend, float(trend_confidence)
