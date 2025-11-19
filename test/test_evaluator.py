"""模型评估器测试"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.evaluator import ModelEvaluator


class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output


@pytest.fixture
def sample_model():
    """创建测试模型"""
    model = SimpleModel(input_size=5, hidden_size=10, output_size=1)
    return model


@pytest.fixture
def sample_data():
    """创建测试数据"""
    np.random.seed(42)
    X = np.random.randn(100, 30, 5).astype(np.float32)  # (样本数, 序列长度, 特征数)
    y = np.random.randn(100).astype(np.float32)
    return X, y


def test_evaluator_initialization(sample_model):
    """测试评估器初始化"""
    evaluator = ModelEvaluator(sample_model)
    assert evaluator.model is not None
    assert evaluator.device in ['cuda', 'cpu']


def test_predict(sample_model, sample_data):
    """测试预测功能"""
    X, _ = sample_data
    evaluator = ModelEvaluator(sample_model)
    
    predictions = evaluator.predict(X, batch_size=16)
    
    assert predictions.shape == (100,)
    assert not np.isnan(predictions).any()


def test_calculate_mae():
    """测试MAE计算"""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    
    mae = ModelEvaluator.calculate_mae(y_true, y_pred)
    
    expected_mae = np.mean(np.abs(y_true - y_pred))
    assert abs(mae - expected_mae) < 1e-6


def test_calculate_rmse():
    """测试RMSE计算"""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    
    rmse = ModelEvaluator.calculate_rmse(y_true, y_pred)
    
    expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    assert abs(rmse - expected_rmse) < 1e-6


def test_calculate_mape():
    """测试MAPE计算"""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([11.0, 22.0, 29.0, 41.0, 48.0])
    
    mape = ModelEvaluator.calculate_mape(y_true, y_pred)
    
    assert 0 <= mape <= 100
    assert mape > 0  # 因为有误差


def test_calculate_mape_with_zero():
    """测试MAPE计算（包含零值）"""
    y_true = np.array([0.0, 20.0, 30.0])
    y_pred = np.array([1.0, 22.0, 29.0])
    
    # 应该跳过零值
    mape = ModelEvaluator.calculate_mape(y_true, y_pred)
    
    assert not np.isnan(mape)
    assert not np.isinf(mape)


def test_calculate_r2():
    """测试R²计算"""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    
    r2 = ModelEvaluator.calculate_r2(y_true, y_pred)
    
    assert -1 <= r2 <= 1  # R²通常在-1到1之间


def test_calculate_direction_accuracy():
    """测试方向准确率计算"""
    # 完全正确的方向预测
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    
    accuracy = ModelEvaluator.calculate_direction_accuracy(y_true, y_pred)
    assert accuracy == 1.0
    
    # 完全错误的方向预测
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    
    accuracy = ModelEvaluator.calculate_direction_accuracy(y_true, y_pred)
    assert accuracy == 0.0


def test_evaluate(sample_model, sample_data):
    """测试完整评估流程"""
    X, y = sample_data
    evaluator = ModelEvaluator(sample_model)
    
    metrics = evaluator.evaluate(X, y, batch_size=16)
    
    # 检查所有指标都存在
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'mape' in metrics
    assert 'r2' in metrics
    assert 'direction_accuracy' in metrics
    assert 'samples' in metrics
    
    # 检查指标值合理
    assert metrics['samples'] == 100
    assert metrics['mae'] >= 0
    assert metrics['rmse'] >= 0
    assert 0 <= metrics['direction_accuracy'] <= 1


def test_generate_report(sample_model, sample_data):
    """测试报告生成"""
    X, y = sample_data
    evaluator = ModelEvaluator(sample_model)
    
    metrics = evaluator.evaluate(X, y)
    report = evaluator.generate_report(metrics)
    
    # 检查报告包含关键信息
    assert '模型评估报告' in report
    assert 'MAE' in report
    assert 'RMSE' in report
    assert 'MAPE' in report
    assert 'R²' in report
    assert '方向准确率' in report


def test_plot_predictions(sample_model, sample_data, tmp_path):
    """测试预测对比图绘制"""
    X, y = sample_data
    evaluator = ModelEvaluator(sample_model)
    
    y_pred = evaluator.predict(X)
    save_path = tmp_path / "predictions.png"
    
    # 测试保存图表
    evaluator.plot_predictions(y, y_pred, save_path=str(save_path))
    
    assert save_path.exists()


def test_plot_loss_curves(sample_model, tmp_path):
    """测试损失曲线绘制"""
    evaluator = ModelEvaluator(sample_model)
    
    train_losses = [0.5, 0.4, 0.3, 0.25, 0.2]
    val_losses = [0.6, 0.5, 0.45, 0.4, 0.38]
    save_path = tmp_path / "loss_curves.png"
    
    evaluator.plot_loss_curves(train_losses, val_losses, save_path=str(save_path))
    
    assert save_path.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
