"""简单的GPU训练测试 - 跳过数据加载，直接测试GPU训练"""
import torch
import torch.nn as nn
import numpy as np
import time

print("=" * 70)
print("  简单GPU训练测试")
print("=" * 70)

# 1. 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("❌ GPU不可用")
    exit(1)

# 2. 创建简单的LSTM模型
print("\n创建模型...")
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=50, hidden_size=256, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = SimpleLSTM(input_size=50, hidden_size=256, num_layers=3).to(device)
print(f"✓ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 3. 创建模拟数据（大批量）
print("\n创建模拟数据...")
batch_size = 256
seq_length = 60
input_size = 50

# 创建大量数据来充分利用GPU
X_train = torch.randn(10000, seq_length, input_size)
y_train = torch.randn(10000, 1)

print(f"✓ 训练数据: {X_train.shape}")
print(f"✓ 批次大小: {batch_size}")

# 4. 训练设置
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 启用混合精度
scaler = torch.cuda.amp.GradScaler()

# 5. 训练循环
print("\n开始训练（10个批次）...")
print("=" * 70)

model.train()
total_batches = 10

# 预热GPU
print("预热GPU...")
for i in range(2):
    idx = torch.randint(0, len(X_train) - batch_size, (1,)).item()
    X_batch = X_train[idx:idx+batch_size].to(device)
    y_batch = y_train[idx:idx+batch_size].to(device)
    
    with torch.cuda.amp.autocast():
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

torch.cuda.synchronize()
print("✓ 预热完成\n")

# 正式训练
start_time = time.time()

for batch_idx in range(total_batches):
    batch_start = time.time()
    
    # 随机采样一个批次
    idx = torch.randint(0, len(X_train) - batch_size, (1,)).item()
    X_batch = X_train[idx:idx+batch_size].to(device)
    y_batch = y_train[idx:idx+batch_size].to(device)
    
    # 混合精度训练
    with torch.cuda.amp.autocast():
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    
    # 同步GPU
    torch.cuda.synchronize()
    
    batch_time = time.time() - batch_start
    
    # 显存使用
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    
    print(f"Batch [{batch_idx+1}/{total_batches}] - "
          f"Loss: {loss.item():.6f}, "
          f"Time: {batch_time:.3f}s, "
          f"显存: {allocated:.2f}GB / {reserved:.2f}GB")

total_time = time.time() - start_time

print("\n" + "=" * 70)
print(f"✅ 训练完成!")
print(f"  总耗时: {total_time:.2f}秒")
print(f"  平均每批: {total_time/total_batches:.3f}秒")
print(f"  最终显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
print("=" * 70)

# 6. GPU性能测试
print("\nGPU性能测试...")
print("测试大矩阵乘法...")

size = 8192
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

torch.cuda.synchronize()
start = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize()
gpu_time = time.time() - start

print(f"✓ GPU计算时间: {gpu_time:.4f}秒")
print(f"✓ 性能: {(size**3 * 2) / gpu_time / 1e12:.2f} TFLOPS")

print("\n" + "=" * 70)
print("如果你看到:")
print("  1. 显存使用 > 2GB")
print("  2. 每批次时间 < 0.1秒")
print("  3. 性能 > 5 TFLOPS")
print("说明GPU工作正常！")
print("\n如果显存仍然很低，问题在于:")
print("  1. 数据加载太慢，GPU在等待数据")
print("  2. batch_size太小")
print("  3. 模型太简单")
print("=" * 70)
