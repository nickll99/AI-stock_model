"""诊断GPU使用情况"""
import torch
import sys
from pathlib import Path

def check_gpu():
    """检查GPU配置"""
    print("=" * 70)
    print("  GPU诊断工具")
    print("=" * 70)
    
    # 1. 检查CUDA是否可用
    print("\n1. CUDA可用性:")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA可用: {cuda_available}")
    
    if not cuda_available:
        print("\n❌ CUDA不可用！")
        print("   可能原因:")
        print("   - PyTorch未安装CUDA版本")
        print("   - CUDA驱动未正确安装")
        print("   - GPU不支持CUDA")
        return False
    
    # 2. GPU信息
    print("\n2. GPU信息:")
    gpu_count = torch.cuda.device_count()
    print(f"   GPU数量: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\n   GPU {i}:")
        print(f"     名称: {torch.cuda.get_device_name(i)}")
        print(f"     计算能力: {props.major}.{props.minor}")
        print(f"     总显存: {props.total_memory / 1024**3:.2f} GB")
        print(f"     多处理器数量: {props.multi_processor_count}")
    
    # 3. 当前显存使用
    print("\n3. 显存使用:")
    for i in range(gpu_count):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = props.total_memory / 1024**3
        print(f"   GPU {i}:")
        print(f"     已分配: {allocated:.2f} GB")
        print(f"     已保留: {reserved:.2f} GB")
        print(f"     使用率: {allocated/total*100:.1f}%")
    
    # 4. PyTorch版本
    print("\n4. PyTorch配置:")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA版本: {torch.version.cuda}")
    print(f"   cuDNN版本: {torch.backends.cudnn.version()}")
    print(f"   cuDNN启用: {torch.backends.cudnn.enabled}")
    
    # 5. 测试GPU性能
    print("\n5. GPU性能测试:")
    print("   创建测试张量...")
    
    try:
        # 创建大张量测试
        size = 10000
        device = torch.device('cuda:0')
        
        # 测试1: 矩阵乘法
        print(f"   测试矩阵乘法 ({size}x{size})...")
        import time
        
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        torch.cuda.synchronize()
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"   ✓ GPU计算时间: {gpu_time:.4f} 秒")
        
        # 测试2: CPU对比
        print(f"   测试CPU对比...")
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        
        start = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start
        
        print(f"   ✓ CPU计算时间: {cpu_time:.4f} 秒")
        print(f"   ✓ GPU加速比: {cpu_time/gpu_time:.1f}x")
        
        # 清理
        del a, b, c, a_cpu, b_cpu, c_cpu
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ❌ 性能测试失败: {e}")
        return False
    
    # 6. 训练建议
    print("\n6. 训练优化建议:")
    
    total_memory = props.total_memory / 1024**3
    
    if total_memory < 4:
        print("   ⚠️  显存较小 (<4GB)")
        print("   建议:")
        print("     - 使用较小的batch_size (32-64)")
        print("     - 使用较小的hidden_size (64-128)")
        print("     - 启用混合精度训练 (--amp)")
    elif total_memory < 8:
        print("   ✓ 显存适中 (4-8GB)")
        print("   建议:")
        print("     - batch_size: 64-128")
        print("     - hidden_size: 128-256")
        print("     - 启用混合精度训练 (--amp)")
    elif total_memory < 16:
        print("   ✓ 显存充足 (8-16GB)")
        print("   建议:")
        print("     - batch_size: 128-256")
        print("     - hidden_size: 256-512")
        print("     - 启用混合精度训练 (--amp)")
    else:
        print("   ✓ 显存丰富 (>16GB)")
        print("   建议:")
        print("     - batch_size: 256-512")
        print("     - hidden_size: 512-1024")
        print("     - 可选择性启用混合精度训练")
    
    print("\n   通用优化建议:")
    print("     1. 启用混合精度训练: --amp")
    print("     2. 增加DataLoader workers: --num-workers 8")
    print("     3. 启用pin_memory: --pin-memory")
    print("     4. 使用更大的batch_size")
    print("     5. 确保数据已预热到缓存")
    
    # 7. 检查当前训练参数
    print("\n7. 当前训练命令分析:")
    if len(sys.argv) > 1:
        cmd = ' '.join(sys.argv[1:])
        print(f"   命令: {cmd}")
        
        # 分析参数
        issues = []
        if '--amp' not in cmd:
            issues.append("❌ 未启用混合精度训练 (--amp)")
        if '--batch-size' in cmd:
            import re
            match = re.search(r'--batch-size\s+(\d+)', cmd)
            if match:
                batch_size = int(match.group(1))
                if batch_size < 128:
                    issues.append(f"⚠️  batch_size较小 ({batch_size})，建议增大到128-256")
        else:
            issues.append("⚠️  未指定batch_size")
        
        if '--num-workers' in cmd:
            match = re.search(r'--num-workers\s+(\d+)', cmd)
            if match:
                num_workers = int(match.group(1))
                if num_workers < 4:
                    issues.append(f"⚠️  num_workers较小 ({num_workers})，建议增大到4-8")
        else:
            issues.append("⚠️  未指定num_workers，使用默认值4")
        
        if issues:
            print("\n   发现的问题:")
            for issue in issues:
                print(f"     {issue}")
        else:
            print("\n   ✓ 参数配置良好")
    
    print("\n" + "=" * 70)
    print("✅ GPU诊断完成")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    check_gpu()
