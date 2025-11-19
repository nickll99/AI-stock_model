"""
GPU检测和配置脚本
检查CUDA是否可用，并提供安装建议
"""
import sys
import subprocess

def check_gpu():
    """检查GPU配置"""
    print("=" * 70)
    print("  GPU配置检查")
    print("=" * 70)
    
    # 1. 检查PyTorch
    print("\n1. 检查PyTorch...")
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch未安装")
        print("\n安装命令:")
        print("  pip install torch torchvision torchaudio")
        return
    
    # 2. 检查CUDA
    print("\n2. 检查CUDA...")
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print(f"✓ CUDA可用: True")
        print(f"✓ CUDA版本: {torch.version.cuda}")
        print(f"✓ GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"✓ GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # GPU内存
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  - 总内存: {total_memory:.2f} GB")
    else:
        print("✗ CUDA不可用")
        print("\n可能的原因:")
        print("  1. 安装的是CPU版本的PyTorch")
        print("  2. 没有NVIDIA GPU")
        print("  3. CUDA驱动未安装或版本不匹配")
    
    # 3. 检查NVIDIA驱动
    print("\n3. 检查NVIDIA驱动...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA驱动已安装")
            print("\nGPU信息:")
            print(result.stdout)
        else:
            print("✗ nvidia-smi命令失败")
    except FileNotFoundError:
        print("✗ nvidia-smi未找到")
        print("  可能没有安装NVIDIA驱动或不是NVIDIA GPU")
    
    # 4. 给出建议
    print("\n" + "=" * 70)
    print("  建议")
    print("=" * 70)
    
    if cuda_available:
        print("\n✓ GPU配置正常，可以使用GPU训练！")
        print("\n使用GPU训练:")
        print("  python examples/train_with_manager.py")
        print("  python scripts/train_universal_model.py --device cuda")
    else:
        print("\n当前使用CPU训练")
        print("\n如果你有NVIDIA GPU，请按以下步骤配置:")
        
        print("\n步骤1：检查是否有NVIDIA GPU")
        print("  Windows: 设备管理器 -> 显示适配器")
        print("  Linux: lspci | grep -i nvidia")
        
        print("\n步骤2：安装NVIDIA驱动")
        print("  访问: https://www.nvidia.com/Download/index.aspx")
        
        print("\n步骤3：安装CUDA版本的PyTorch")
        print("  # 卸载当前PyTorch")
        print("  pip uninstall torch torchvision torchaudio")
        print("")
        print("  # 安装CUDA 11.8版本")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("")
        print("  # 或安装CUDA 12.1版本")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        print("\n步骤4：重新运行此脚本验证")
        print("  python scripts/check_gpu.py")
    
    # 5. 性能对比
    print("\n" + "=" * 70)
    print("  性能对比")
    print("=" * 70)
    
    print("\n训练5000只股票:")
    print("  CPU (8核): 40-50小时")
    print("  GPU (GTX 1660): 15-20小时")
    print("  GPU (RTX 3080): 8-12小时")
    print("  GPU (RTX 4090): 4-6小时")
    
    print("\n训练通用模型:")
    print("  CPU (8核): 5小时")
    print("  GPU (GTX 1660): 2小时")
    print("  GPU (RTX 3080): 1小时")
    print("  GPU (RTX 4090): 30分钟")
    
    print("\n" + "=" * 70)
    print()


if __name__ == "__main__":
    check_gpu()
