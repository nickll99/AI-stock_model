"""
通用股票预测模型
一个模型预测所有股票
"""
import torch
import torch.nn as nn
from typing import Optional


class UniversalStockModel(nn.Module):
    """
    通用股票预测模型
    
    特点：
    1. 使用股票Embedding来区分不同股票
    2. 一个模型可以预测所有股票
    3. 利用全市场数据训练，泛化能力强
    """
    
    def __init__(
        self,
        num_stocks: int,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        stock_embedding_dim: int = 32,
        model_type: str = 'lstm'
    ):
        """
        初始化通用模型
        
        Args:
            num_stocks: 股票总数
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: 层数
            dropout: Dropout比例
            stock_embedding_dim: 股票嵌入维度
            model_type: 模型类型 (lstm/gru/transformer)
        """
        super(UniversalStockModel, self).__init__()
        
        self.num_stocks = num_stocks
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        self.stock_embedding_dim = stock_embedding_dim
        
        # 股票Embedding层
        # 将股票代码映射为向量表示
        self.stock_embedding = nn.Embedding(num_stocks, stock_embedding_dim)
        
        # 组合输入：原始特征 + 股票embedding
        combined_input_size = input_size + stock_embedding_dim
        
        # 时序模型
        if model_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=combined_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif model_type == 'gru':
            self.rnn = nn.GRU(
                input_size=combined_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, stock_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_length, input_size]
            stock_ids: 股票ID [batch_size]
            
        Returns:
            预测值 [batch_size, 1]
        """
        batch_size, seq_length, _ = x.size()
        
        # 获取股票embedding [batch_size, stock_embedding_dim]
        stock_emb = self.stock_embedding(stock_ids)
        
        # 扩展到序列长度 [batch_size, seq_length, stock_embedding_dim]
        stock_emb = stock_emb.unsqueeze(1).expand(batch_size, seq_length, -1)
        
        # 拼接原始特征和股票embedding
        # [batch_size, seq_length, input_size + stock_embedding_dim]
        x_combined = torch.cat([x, stock_emb], dim=2)
        
        # RNN处理
        if self.model_type in ['lstm', 'gru']:
            out, _ = self.rnn(x_combined)
            # 取最后一个时间步的输出
            out = out[:, -1, :]  # [batch_size, hidden_size]
        
        # 全连接层
        out = self.fc(out)  # [batch_size, 1]
        
        return out
    
    def get_stock_embedding(self, stock_id: int) -> torch.Tensor:
        """
        获取指定股票的embedding向量
        
        Args:
            stock_id: 股票ID
            
        Returns:
            股票embedding向量
        """
        stock_id_tensor = torch.tensor([stock_id], dtype=torch.long)
        return self.stock_embedding(stock_id_tensor).detach()


class UniversalTransformerModel(nn.Module):
    """
    基于Transformer的通用股票预测模型
    """
    
    def __init__(
        self,
        num_stocks: int,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        stock_embedding_dim: int = 32
    ):
        """
        初始化Transformer通用模型
        
        Args:
            num_stocks: 股票总数
            input_size: 输入特征维度
            d_model: Transformer模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dropout: Dropout比例
            stock_embedding_dim: 股票嵌入维度
        """
        super(UniversalTransformerModel, self).__init__()
        
        self.num_stocks = num_stocks
        self.input_size = input_size
        self.d_model = d_model
        self.stock_embedding_dim = stock_embedding_dim
        
        # 股票Embedding
        self.stock_embedding = nn.Embedding(num_stocks, stock_embedding_dim)
        
        # 输入投影
        combined_input_size = input_size + stock_embedding_dim
        self.input_projection = nn.Linear(combined_input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, stock_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_length, input_size]
            stock_ids: 股票ID [batch_size]
            
        Returns:
            预测值 [batch_size, 1]
        """
        batch_size, seq_length, _ = x.size()
        
        # 获取股票embedding
        stock_emb = self.stock_embedding(stock_ids)
        stock_emb = stock_emb.unsqueeze(1).expand(batch_size, seq_length, -1)
        
        # 拼接特征
        x_combined = torch.cat([x, stock_emb], dim=2)
        
        # 输入投影
        x_proj = self.input_projection(x_combined)
        
        # 位置编码
        x_pos = self.pos_encoder(x_proj)
        
        # Transformer编码
        out = self.transformer_encoder(x_pos)
        
        # 取最后一个时间步
        out = out[:, -1, :]
        
        # 输出层
        out = self.fc(out)
        
        return out


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_length, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
