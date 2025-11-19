"""模型训练器"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Optional, Callable
from pathlib import Path
import time


class ModelTrainer:
    """模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        初始化训练器
        
        Args:
            model: PyTorch模型
            device: 训练设备
            checkpoint_dir: 检查点保存目录
        """
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        learning_rate: float = 0.001,
        patience: int = 10,
        checkpoint_interval: int = 10,
        scheduler_type: str = 'reduce_on_plateau',
        callback: Optional[Callable] = None
    ) -> Dict:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
            patience: 早停耐心值
            checkpoint_interval: 检查点保存间隔
            scheduler_type: 学习率调度器类型
            callback: 回调函数（用于报告进度）
            
        Returns:
            训练历史字典
        """
        # 优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 损失函数
        criterion = nn.MSELoss()
        
        # 学习率调度器
        if scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5
            )
        else:
            scheduler = None
        
        print(f"开始训练，设备: {self.device}")
        print(f"训练样本数: {len(train_loader.dataset)}, 验证样本数: {len(val_loader.dataset)}")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 训练阶段
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            
            # 验证阶段
            val_loss = self._validate_epoch(val_loader, criterion)
            self.val_losses.append(val_loss)
            
            # 学习率调度
            if scheduler:
                if scheduler_type == 'reduce_on_plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            
            # 打印进度
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"Time: {epoch_time:.2f}s")
            
            # 回调函数
            if callback:
                callback(epoch + 1, train_loss, val_loss)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch + 1, {'val_loss': val_loss}, is_best=True)
            else:
                self.patience_counter += 1
            
            # 定期保存检查点
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1, {'val_loss': val_loss})
            
            # 早停
            if self.patience_counter >= patience:
                print(f"早停触发，在第 {epoch + 1} 轮停止训练")
                break
        
        print("训练完成！")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_epochs': epoch + 1
        }
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            
            # 确保输出和目标形状一致
            if outputs.dim() == 2 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                
                if outputs.dim() == 2 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict,
        is_best: bool = False
    ) -> str:
        """
        保存模型检查点
        
        Args:
            epoch: 当前轮数
            metrics: 评估指标
            is_best: 是否为最佳模型
            
        Returns:
            检查点文件路径
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        加载模型检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            检查点字典
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        print(f"检查点已加载: {checkpoint_path}")
        return checkpoint
    
    @staticmethod
    def create_data_loader(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> DataLoader:
        """
        创建数据加载器
        
        Args:
            X: 特征数据
            y: 目标数据
            batch_size: 批次大小
            shuffle: 是否打乱
            
        Returns:
            DataLoader对象
        """
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return loader
