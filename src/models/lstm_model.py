"""LSTM预测模型"""
import torch
import torch.nn as nn
from typing import Optional


class LSTMModel(nn.Module):
    """LSTM预测模型"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            output_size: 输出维度
            dropout: Dropout比例
            bidirectional: 是否使用双向LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 全连接层
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_length, input_size)
            
        Returns:
            输出张量 (batch_size, output_size)
        """
        # LSTM层
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最后一个时间步的输出
        if self.bidirectional:
            # 双向LSTM：拼接前向和后向的最后隐藏状态
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            last_output = torch.cat((h_forward, h_backward), dim=1)
        else:
            # 单向LSTM：取最后一层的隐藏状态
            last_output = h_n[-1, :, :]
        
        # 全连接层
        output = self.fc(last_output)
        
        return output
    
    def init_hidden(self, batch_size: int, device: torch.device) -> tuple:
        """
        初始化隐藏状态
        
        Args:
            batch_size: 批次大小
            device: 设备
            
        Returns:
            (h_0, c_0) 元组
        """
        num_directions = 2 if self.bidirectional else 1
        h_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size
        ).to(device)
        c_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size
        ).to(device)
        return h_0, c_0


class LSTMModelConfig:
    """LSTM模型配置"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        seq_length: int = 60
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.seq_length = seq_length
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'seq_length': self.seq_length
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'LSTMModelConfig':
        """从字典创建配置"""
        return cls(**config_dict)
