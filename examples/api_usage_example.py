"""API使用示例 - 展示如何使用请求追踪ID和处理响应"""
import requests
import json
from datetime import datetime

# API基础URL
BASE_URL = "http://localhost:8000"


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_response_info(response):
    """打印响应信息"""
    print(f"\n状态码: {response.status_code}")
    print(f"请求追踪ID: {response.headers.get('X-Request-ID', 'N/A')}")
    print(f"处理时间: {response.headers.get('X-Process-Time', 'N/A')}秒")
    print(f"限流 - 剩余: {response.headers.get('X-RateLimit-Remaining', 'N/A')}/{response.headers.get('X-RateLimit-Limit', 'N/A')}")


def example_1_health_check():
    """示例1: 健康检查"""
    print_section("示例1: 健康检查")
    
    response = requests.get(f"{BASE_URL}/health")
    print_response_info(response)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n服务状态: {data['status']}")
        print(f"服务名称: {data['service']}")
        print(f"版本: {data['version']}")


def example_2_custom_request_id():
    """示例2: 使用自定义请求ID"""
    print_section("示例2: 使用自定义请求ID")
    
    custom_id = f"custom-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    headers = {"X-Request-ID": custom_id}
    
    print(f"\n发送自定义请求ID: {custom_id}")
    
    response = requests.get(f"{BASE_URL}/health", headers=headers)
    print_response_info(response)
    
    returned_id = response.headers.get('X-Request-ID')
    print(f"\n返回的请求ID: {returned_id}")
    print(f"ID匹配: {'✓' if returned_id == custom_id else '✗'}")


def example_3_get_stocks():
    """示例3: 获取股票列表"""
    print_section("示例3: 获取股票列表")
    
    params = {
        "market": "主板",
        "limit": 5
    }
    
    print(f"\n请求参数: {params}")
    
    response = requests.get(f"{BASE_URL}/api/v1/data/stocks", params=params)
    print_response_info(response)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n返回股票数量: {data['count']}")
        print("\n股票列表:")
        for stock in data['stocks'][:3]:  # 只显示前3个
            print(f"  - {stock['symbol']}: {stock['name']}")


def example_4_error_handling():
    """示例4: 错误处理"""
    print_section("示例4: 错误处理")
    
    # 请求不存在的股票
    symbol = "INVALID"
    print(f"\n请求不存在的股票: {symbol}")
    
    response = requests.get(f"{BASE_URL}/api/v1/data/stocks/{symbol}/info")
    print_response_info(response)
    
    if response.status_code != 200:
        error_data = response.json()
        print("\n错误信息:")
        print(f"  错误代码: {error_data['error']['code']}")
        print(f"  错误消息: {error_data['error']['message']}")
        print(f"  时间戳: {error_data['error']['timestamp']}")
        print(f"  请求ID: {error_data['error']['request_id']}")


def example_5_rate_limiting():
    """示例5: 测试限流"""
    print_section("示例5: 测试限流（发送多个请求）")
    
    print("\n发送10个连续请求...")
    
    for i in range(10):
        response = requests.get(f"{BASE_URL}/health")
        remaining = response.headers.get('X-RateLimit-Remaining', 'N/A')
        print(f"请求 {i+1}: 状态码={response.status_code}, 剩余请求数={remaining}")
        
        if response.status_code == 429:
            print("\n触发限流！")
            error_data = response.json()
            print(f"错误消息: {error_data['error']['message']}")
            break


def example_6_prediction_request():
    """示例6: 预测请求（需要模型已训练）"""
    print_section("示例6: 预测请求")
    
    payload = {
        "stock_code": "000001",
        "days": 5,
        "model_version": "latest"
    }
    
    print(f"\n请求体: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    response = requests.post(
        f"{BASE_URL}/api/v1/prediction/predict",
        json=payload
    )
    print_response_info(response)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n股票代码: {data.get('stock_code')}")
        print(f"趋势: {data.get('trend')}")
        print(f"置信度: {data.get('confidence_score')}")
        
        if 'predictions' in data and len(data['predictions']) > 0:
            print("\n预测结果（前3天）:")
            for pred in data['predictions'][:3]:
                print(f"  {pred['date']}: {pred['price']} "
                      f"[{pred['confidence_lower']}-{pred['confidence_upper']}]")
    else:
        print("\n注意: 此示例需要先训练模型")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  A股AI预测系统 - API使用示例")
    print("=" * 60)
    print("\n确保API服务已启动: python -m uvicorn src.api.main:app --reload")
    print("访问文档: http://localhost:8000/docs")
    
    try:
        # 示例1: 健康检查
        example_1_health_check()
        
        # 示例2: 自定义请求ID
        example_2_custom_request_id()
        
        # 示例3: 获取股票列表
        example_3_get_stocks()
        
        # 示例4: 错误处理
        example_4_error_handling()
        
        # 示例5: 限流测试
        example_5_rate_limiting()
        
        # 示例6: 预测请求（可选）
        # example_6_prediction_request()
        
        print("\n" + "=" * 60)
        print("  示例完成")
        print("=" * 60)
        print("\n提示:")
        print("- 所有响应都包含 X-Request-ID 用于追踪")
        print("- 使用 X-Process-Time 监控性能")
        print("- 注意 X-RateLimit-* 响应头避免限流")
        print("- 错误响应包含详细的错误信息和请求ID")
        print()
        
    except requests.exceptions.ConnectionError:
        print("\n" + "=" * 60)
        print("  错误: 无法连接到API服务")
        print("=" * 60)
        print("\n请先启动API服务:")
        print("  python -m uvicorn src.api.main:app --reload")
        print()


if __name__ == "__main__":
    main()
