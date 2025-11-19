"""
诊断GPU配置和PyTorch环境

使用方法:
    python diagnose_gpu.py
"""
import torch
import sys
import platform

print("=" * 70)
print("  PyTorch GPU 诊断工具")
print("=" * 70)

# 系统信息
print(f"\n系统信息:")
print(f"  操作系统: {platform.system()} {platform.release()}")
print(f"  Python版本: {sys.version.split()[0]}")

# PyTorch版本
print(f"\nPyTorch信息:")
print(f"  PyTorch版本: {torch.__version__}")
print(f"  安装路径: {torch.__file__}")

# 检查CUDA/ROCm
cuda_available = torch.cuda.is_available()
print(f"\nGPU支持:")
print(f"  CUDA/ROCm可用: {'✅ 是' if cuda_available else '❌ 否'}")

if cuda_available:
    # 设备信息
    device_count = torch.cuda.device_count()
    print(f"  GPU数量: {device_count}")
    
    for i in range(device_count):
        print(f"\n  GPU {i}:")
        print(f"    名称: {torch.cuda.get_device_name(i)}")
        
        props = torch.cuda.get_device_properties(i)
        print(f"    显存总量: {props.total_memory / 1024**3:.2f} GB")
        print(f"    计算能力: {props.major}.{props.minor}")
        print(f"    多处理器数量: {props.multi_processor_count}")
    
    # 当前显存使用
    print(f"\n  当前显存使用:")
    for i in range(device_count):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"    GPU {i}: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")
    
    # 测试GPU计算
    print(f"\nGPU计算测试:")
    try:
        print("  测试1: 创建张量...")
        x = torch.randn(1000, 1000).cuda()
        print("  ✅ 成功")
        
        print("  测试2: 矩阵乘法...")
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("  ✅ 成功")
        
        print("  测试3: 梯度计算...")
        x.requires_grad = True
        y.requires_grad = True
        z = (x * y).sum()
        z.backward()
        print("  ✅ 成功")
        
        print("\n✅ 所有GPU计算测试通过！")
        
    except Exception as e:
        print(f"\n❌ GPU计算测试失败: {e}")
        print("\n可能的原因:")
        print("  1. GPU驱动问题")
        print("  2. PyTorch版本不兼容")
        print("  3. 显存不足")
else:
    print("\n❌ CUDA/ROCm不可用！")
    print("\n可能的原因:")
    print("  1. PyTorch没有安装GPU版本")
    print("  2. GPU驱动未正确安装")
    print("  3. 环境变量未正确设置")
    
    print("\n解决方案:")
    print("\n  对于NVIDIA GPU (CUDA):")
    print("    pip uninstall torch torchvision torchaudio -y")
    print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n  对于AMD GPU (ROCm):")
    print("    pip uninstall torch torchvision torchaudio -y")
    print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6")

# 混合精度支持
print(f"\n混合精度训练:")
if cuda_available:
    try:
        from torch.cuda.amp import autocast, GradScaler
        print("  ✅ 支持混合精度训练（AMP）")
        
        # 测试混合精度
        print("  测试混合精度计算...")
        scaler = GradScaler()
        with autocast():
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
        print("  ✅ 混合精度测试成功")
    except Exception as e:
        print(f"  ⚠️  混合精度可能不完全支持: {e}")
else:
    print("  ❌ 需要GPU支持")

# 推荐配置
print("\n" + "=" * 70)
print("  推荐训练配置")
print("=" * 70)

if cuda_available:
    device_name = torch.cuda.get_device_name(0).lower()
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"\n根据你的GPU配置 ({torch.cuda.get_device_name(0)}, {total_memory:.0f}GB):")
    
    if total_memory >= 16:
        print("\n  高性能配置:")
        print("    python scripts/train_universal_model.py \\")
        print("        --device cuda \\")
        print("        --batch-size 512 \\")
        print("        --hidden-size 256 \\")
        print("        --amp \\")
        print("        --num-workers 0")
    elif total_memory >= 12:
        print("\n  平衡配置:")
        print("    python scripts/train_universal_model.py \\")
        print("        --device cuda \\")
        print("        --batch-size 256 \\")
        print("        --hidden-size 128 \\")
        print("        --amp \\")
        print("        --num-workers 0")
    elif total_memory >= 8:
        print("\n  内存优化配置:")
        print("    python scripts/train_universal_model.py \\")
        print("        --device cuda \\")
        print("        --batch-size 128 \\")
        print("        --hidden-size 64 \\")
        print("        --num-workers 0 \\")
        print("        --limit 500")
    else:
        print("\n  低显存配置:")
        print("    python scripts/train_universal_model.py \\")
        print("        --device cuda \\")
        print("        --batch-size 64 \\")
        print("        --hidden-size 64 \\")
        print("        --num-workers 0 \\")
        print("        --limit 200")
    
    # AMD GPU特殊说明
    if 'amd' in device_name or 'radeon' in device_name:
        print("\n  ⚠️  检测到AMD GPU，建议:")
        print("    1. 使用 --num-workers 0 避免内存问题")
        print("    2. 如果混合精度报错，移除 --amp 参数")
        print("    3. 使用 rocm-smi 监控GPU使用")
else:
    print("\n  请先解决GPU不可用的问题")

print("\n" + "=" * 70)
print("  诊断完成")
print("=" * 70)
print()
