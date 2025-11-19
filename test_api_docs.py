"""测试API文档和日志功能"""
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logging, get_logger

def test_logging():
    """测试日志功能"""
    print("=" * 60)
    print("测试日志功能")
    print("=" * 60)
    
    # 配置日志
    setup_logging(
        log_level="INFO",
        log_file="logs/test.log",
        json_format=True
    )
    
    # 获取日志器
    logger = get_logger(__name__)
    
    # 测试不同级别的日志
    logger.debug("这是一条DEBUG日志")
    logger.info("这是一条INFO日志")
    logger.warning("这是一条WARNING日志")
    logger.error("这是一条ERROR日志")
    
    # 测试带额外字段的日志
    logger.info("带请求ID的日志", extra={"request_id": "test-123", "user_id": "user-456"})
    
    print("\n✓ 日志测试完成，请查看 logs/test.log 文件")
    print("✓ 控制台输出应该是JSON格式\n")


def test_api_docs():
    """测试API文档"""
    print("=" * 60)
    print("测试API文档")
    print("=" * 60)
    
    try:
        from src.api.main import app
        
        # 获取OpenAPI schema
        openapi_schema = app.openapi()
        
        print(f"\n✓ API标题: {openapi_schema['info']['title']}")
        print(f"✓ API版本: {openapi_schema['info']['version']}")
        print(f"✓ API描述长度: {len(openapi_schema['info']['description'])} 字符")
        
        # 检查标签
        if 'tags' in openapi_schema:
            print(f"\n✓ API标签数量: {len(openapi_schema['tags'])}")
            for tag in openapi_schema['tags']:
                print(f"  - {tag['name']}: {tag['description']}")
        
        # 检查自定义响应头
        if 'components' in openapi_schema and 'headers' in openapi_schema['components']:
            print(f"\n✓ 自定义响应头数量: {len(openapi_schema['components']['headers'])}")
            for header_name in openapi_schema['components']['headers']:
                print(f"  - {header_name}")
        
        # 检查路由数量
        path_count = len(openapi_schema['paths'])
        print(f"\n✓ API路由数量: {path_count}")
        
        print("\n✓ API文档配置正确")
        print("✓ 启动服务后访问 http://localhost:8000/docs 查看完整文档\n")
        
    except Exception as e:
        print(f"\n✗ API文档测试失败: {e}\n")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("API文档和日志功能测试")
    print("=" * 60 + "\n")
    
    # 测试日志
    test_logging()
    
    # 测试API文档
    test_api_docs()
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n下一步：")
    print("1. 启动API服务: python -m uvicorn src.api.main:app --reload")
    print("2. 访问文档: http://localhost:8000/docs")
    print("3. 查看日志: tail -f logs/app.log")
    print()


if __name__ == "__main__":
    main()
