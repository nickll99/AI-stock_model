"""
快速测试GPU训练是否工作

使用方法:
    python test_gpu_training.py
"""
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader
import numpy as np

print("=" * 70)
print("  GPU训练快速测试")
print("=" * 70)

# 检查GPU
print(f"\n1. 检查GPU...")
cuda_available = torch.cuda.is_available()
print(f"   CUDA可用: {cuda_available}")

if not cuda_available:
    print("\n❌ GPU不可用，请检查PyTorch安装")
    exit(1)

device = torch.device('cuda')
print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
print(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 创建简单的数据集
print(f"\n2. 创建测试数据...")
class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.X = np.random.randn(size, 60, 48).astype(np.float32)
        self.y = np.random.randn(size, 1).astype(np.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])

dataset = SimpleDataset(size=1000)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
print(f"   ✓ 数据集大小: {len(dataset)}")
print(f"   ✓ Batch size: 128")

# 创建简单的模型
print(f"\n3. 创建模型...")
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(48, 128, 2, batch_first=True)
        self.fc = nn.Linear(128, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = SimpleModel().to(device)
print(f"   ✓ 模型已创建并移动到GPU")

# 训练
print(f"\n4. 开始训练...")
print(f"   {'='*66}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 混合精度
use_amp = True
scaler = torch.cuda.amp.GradScaler() if use_amp else None
print(f"   混合精度: {'启用' if use_amp else '禁用'}")

# 训练5个epoch
for epoch in range(5):
    model.train()
    total_loss = 0
    epoch_start = time.time()
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / len(dataloader)
    
    print(f"   Epoch [{epoch+1}/5] - Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")

print(f"   {'='*66}")

# 检查GPU使用
print(f"\n5. GPU使用情况:")
print(f"   显存已分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"   显存已保留: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

print(f"\n{'='*70}")
print(f"  测试完成！")
print(f"{'='*70}")

print(f"\n✅ GPU训练正常工作！")
print(f"\n如果你看到:")
print(f"  1. 每个epoch时间在1-3秒")
print(f"  2. 显存有分配（>0 GB）")
print(f"  3. 没有错误信息")
print(f"\n说明GPU配置正确，可以开始正式训练了！")

print(f"\n推荐命令:")
print(f"  # 快速测试（100只股票）")
print(f"  python scripts/train_universal_model.py --device cuda --amp --limit 100 --epochs 5")
print(f"\n  # 正式训练（1000只股票）")
print(f"  python scripts/train_universal_model.py --device cuda --amp --limit 1000 --batch-size 1024")
print()
