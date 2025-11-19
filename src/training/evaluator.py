"""模型评估器"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化评估器
        
        Args:
            model: PyTorch模型
            device: 评估设备
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        batch_size: int = 32
    ) -> Dict:
        """
        评估模型性能
        
        Args:
            X: 特征数据
            y_true: 真实目标值
            batch_size: 批次大小
            
        Returns:
            评估指标字典 (MAE, RMSE, MAPE, 方向准确率等)
        """
        # 获取预测值
        y_pred = self.predict(X, batch_size)
        
        # 计算各项指标
        mae = self.calculate_mae(y_true, y_pred)
        rmse = self.calculate_rmse(y_true, y_pred)
        mape = self.calculate_mape(y_true, y_pred)
        r2 = self.calculate_r2(y_true, y_pred)
        direction_accuracy = self.calculate_direction_accuracy(y_true, y_pred)
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'direction_accuracy': float(direction_accuracy),
            'samples': len(y_true)
        }
        
        return metrics
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        批量预测
        
        Args:
            X: 特征数据
            batch_size: 批次大小
            
        Returns:
            预测值数组
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = torch.FloatTensor(X[i:i + batch_size]).to(self.device)
                batch_pred = self.model(batch_X)
                
                if batch_pred.dim() == 2 and batch_pred.size(1) == 1:
                    batch_pred = batch_pred.squeeze(1)
                
                predictions.append(batch_pred.cpu().numpy())
        
        return np.concatenate(predictions)
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算平均绝对误差 (MAE)"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算均方根误差 (RMSE)"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算平均绝对百分比误差 (MAPE)"""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算R²分数"""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def calculate_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算方向准确率（预测涨跌方向的准确率）
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            方向准确率 (0-1)
        """
        if len(y_true) < 2:
            return 0.0
        
        # 计算真实和预测的变化方向
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        # 计算方向一致的比例
        correct = np.sum(true_direction == pred_direction)
        total = len(true_direction)
        
        return correct / total if total > 0 else 0.0
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: list = None,
        save_path: str = None,
        title: str = "预测结果对比"
    ) -> None:
        """
        绘制预测结果对比图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            dates: 日期列表
            save_path: 保存路径
            title: 图表标题
        """
        plt.figure(figsize=(15, 6))
        
        x_axis = dates if dates else range(len(y_true))
        
        plt.plot(x_axis, y_true, label='真实值', color='blue', linewidth=1.5)
        plt.plot(x_axis, y_pred, label='预测值', color='red', linewidth=1.5, alpha=0.7)
        
        plt.xlabel('时间')
        plt.ylabel('价格')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_loss_curves(
        self,
        train_losses: list,
        val_losses: list,
        save_path: str = None
    ) -> None:
        """
        绘制损失曲线
        
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, label='训练损失', color='blue', linewidth=2)
        plt.plot(epochs, val_losses, label='验证损失', color='red', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练和验证损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"损失曲线已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(
        self,
        metrics: Dict,
        save_path: str = None
    ) -> str:
        """
        生成评估报告
        
        Args:
            metrics: 评估指标字典
            save_path: 保存路径
            
        Returns:
            报告文本
        """
        report = "=" * 50 + "\n"
        report += "模型评估报告\n"
        report += "=" * 50 + "\n\n"
        
        report += f"样本数量: {metrics.get('samples', 'N/A')}\n\n"
        
        report += "性能指标:\n"
        report += "-" * 50 + "\n"
        report += f"MAE (平均绝对误差):        {metrics.get('mae', 0):.6f}\n"
        report += f"RMSE (均方根误差):         {metrics.get('rmse', 0):.6f}\n"
        report += f"MAPE (平均绝对百分比误差): {metrics.get('mape', 0):.2f}%\n"
        report += f"R² (决定系数):             {metrics.get('r2', 0):.6f}\n"
        report += f"方向准确率:                {metrics.get('direction_accuracy', 0):.2%}\n"
        
        report += "\n" + "=" * 50 + "\n"
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"评估报告已保存: {save_path}")
        
        return report
