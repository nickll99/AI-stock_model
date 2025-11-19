"""
配置检查脚本
检查系统配置是否正确
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def check_env_file():
    """检查.env文件"""
    print("=" * 60)
    print("  检查环境配置文件")
    print("=" * 60)
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        print("✗ .env 文件不存在")
        if env_example.exists():
            print("  建议: copy .env.example .env")
        return False
    
    print("✓ .env 文件存在")
    
    # 读取配置
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_keys = [
            'DB_HOST', 'DB_PORT', 'DB_USER', 'DB_PASSWORD', 'DB_NAME'
        ]
        
        missing_keys = []
        for key in required_keys:
            if key not in content:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"✗ 缺少必需配置: {', '.join(missing_keys)}")
            return False
        
        print("✓ 必需配置项完整")
        return True
        
    except Exception as e:
        print(f"✗ 读取.env文件失败: {e}")
        return False


def check_config_import():
    """检查配置导入"""
    print("\n" + "=" * 60)
    print("  检查配置导入")
    print("=" * 60)
    
    try:
        from src.config import settings
        
        print("✓ 配置模块导入成功")
        print(f"\n当前配置:")
        print(f"  数据库主机: {settings.DB_HOST}")
        print(f"  数据库端口: {settings.DB_PORT}")
        print(f"  数据库名称: {settings.DB_NAME}")
        print(f"  数据库用户: {settings.DB_USER}")
        print(f"  应用环境: {settings.APP_ENV}")
        print(f"  日志级别: {settings.LOG_LEVEL}")
        
        # 检查数据库URL
        db_url = settings.database_url
        print(f"\n✓ 数据库连接URL: {db_url.replace(settings.DB_PASSWORD, '***')}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_database_connection():
    """检查数据库连接"""
    print("\n" + "=" * 60)
    print("  检查数据库连接")
    print("=" * 60)
    
    try:
        from src.database.connection import engine, get_db_context
        from src.database.models import StockBasicInfo, StockKlineData
        
        print("\n尝试连接数据库...")
        
        # 测试连接
        with engine.connect() as conn:
            print("✓ 数据库连接成功")
        
        # 检查表是否存在
        print("\n检查数据表...")
        with get_db_context() as db:
            # 检查stock_basic_info表
            try:
                count = db.query(StockBasicInfo).count()
                print(f"✓ stock_basic_info 表存在，记录数: {count}")
            except Exception as e:
                print(f"✗ stock_basic_info 表不存在或无法访问: {e}")
            
            # 检查stock_kline_data表
            try:
                count = db.query(StockKlineData).count()
                print(f"✓ stock_kline_data 表存在，记录数: {count}")
            except Exception as e:
                print(f"✗ stock_kline_data 表不存在或无法访问: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据库连接失败: {e}")
        print("\n请检查:")
        print("  1. MySQL服务是否启动")
        print("  2. .env文件中的数据库配置是否正确")
        print("  3. 数据库是否已创建")
        print("  4. 用户是否有访问权限")
        return False


def check_redis_usage():
    """检查Redis使用情况"""
    print("\n" + "=" * 60)
    print("  检查Redis使用情况")
    print("=" * 60)
    
    # 检查训练模块是否使用Redis
    training_files = [
        'src/training/trainer.py',
        'src/training/evaluator.py',
        'src/data/loader.py',
        'src/features/engineer.py',
        'src/features/dataset_builder.py'
    ]
    
    redis_found = False
    for file_path in training_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'redis' in content.lower() or 'cache_manager' in content:
                    print(f"⚠ {file_path} 中发现Redis使用")
                    redis_found = True
        except FileNotFoundError:
            pass
    
    if not redis_found:
        print("✓ 训练模块未使用Redis缓存（符合日线级别训练要求）")
    else:
        print("✗ 训练模块不应使用Redis缓存")
    
    # 检查预测模块是否使用Redis
    print("\n检查预测模块Redis使用...")
    try:
        with open('src/prediction/engine.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'redis' in content.lower() or 'cache' in content.lower():
                print("✓ 预测模块可以使用Redis缓存")
            else:
                print("  预测模块未使用Redis缓存")
    except FileNotFoundError:
        pass
    
    return not redis_found


def check_gitignore():
    """检查.gitignore文件"""
    print("\n" + "=" * 60)
    print("  检查.gitignore文件")
    print("=" * 60)
    
    gitignore_file = Path(".gitignore")
    
    if not gitignore_file.exists():
        print("✗ .gitignore 文件不存在")
        return False
    
    print("✓ .gitignore 文件存在")
    
    # 检查关键忽略项
    with open(gitignore_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_ignores = [
        'logs/',
        'checkpoints/',
        '.env',
        '__pycache__/',
        'venv/',
        '*.pth'
    ]
    
    missing = []
    for item in required_ignores:
        if item not in content:
            missing.append(item)
    
    if missing:
        print(f"⚠ 建议添加以下忽略项: {', '.join(missing)}")
    else:
        print("✓ 关键忽略项已配置")
    
    return True


def check_directories():
    """检查必要的目录"""
    print("\n" + "=" * 60)
    print("  检查目录结构")
    print("=" * 60)
    
    required_dirs = [
        'src/data',
        'src/features',
        'src/models',
        'src/training',
        'src/prediction',
        'src/database',
        'src/api',
        'docs',
        'examples'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} 不存在")
            all_exist = False
    
    # 检查需要创建的目录
    create_dirs = ['logs', 'checkpoints']
    print("\n检查运行时目录...")
    for dir_path in create_dirs:
        path = Path(dir_path)
        if not path.exists():
            print(f"  {dir_path} 将在运行时自动创建")
        else:
            print(f"✓ {dir_path}")
    
    return all_exist


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  A股AI模型系统 - 配置检查")
    print("=" * 60)
    
    results = []
    
    # 检查1: 环境配置文件
    results.append(("环境配置文件", check_env_file()))
    
    # 检查2: 配置导入
    results.append(("配置导入", check_config_import()))
    
    # 检查3: 数据库连接
    results.append(("数据库连接", check_database_connection()))
    
    # 检查4: Redis使用
    results.append(("Redis使用检查", check_redis_usage()))
    
    # 检查5: .gitignore
    results.append(("Git忽略文件", check_gitignore()))
    
    # 检查6: 目录结构
    results.append(("目录结构", check_directories()))
    
    # 总结
    print("\n" + "=" * 60)
    print("  检查总结")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    print(f"\n总计: {success_count}/{total_count} 检查通过")
    
    if success_count == total_count:
        print("\n✓ 所有检查通过！系统配置正确。")
        print("\n下一步:")
        print("  1. 运行快速测试: python quick_test.py")
        print("  2. 初始化AI表: python src/database/init_db.py")
        print("  3. 运行完整测试: python test_training_prediction.py")
    else:
        print("\n⚠ 部分检查失败，请根据提示修复问题。")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
