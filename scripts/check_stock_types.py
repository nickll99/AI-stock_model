"""
查看股票类型分布

使用方法:
    python scripts/check_stock_types.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import StockDataLoader


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("  股票类型分布统计")
    print("=" * 70)
    
    loader = StockDataLoader()
    
    # 常见股票类型
    stock_types = ['主板', '创业板', '科创板', '中小板', '北交所']
    
    print("\n按类型统计:")
    print("-" * 70)
    print(f"{'股票类型':<15} {'数量':>10} {'占比':>10}")
    print("-" * 70)
    
    type_counts = {}
    total = 0
    
    for stock_type in stock_types:
        symbols = loader.get_all_active_stocks(stock_type=stock_type)
        count = len(symbols)
        type_counts[stock_type] = count
        total += count
    
    # 显示各类型统计
    for stock_type, count in type_counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{stock_type:<15} {count:>10} {percentage:>9.1f}%")
    
    print("-" * 70)
    print(f"{'小计':<15} {total:>10} {100.0:>9.1f}%")
    
    # 查询所有活跃股票
    all_symbols = loader.get_all_active_stocks()
    print(f"{'全部活跃股票':<15} {len(all_symbols):>10}")
    
    # 如果总数不一致，说明有其他类型
    if len(all_symbols) != total:
        other_count = len(all_symbols) - total
        print(f"{'其他类型':<15} {other_count:>10}")
    
    print("=" * 70)
    
    # 显示示例股票
    print("\n示例股票（每种类型前5只）:")
    print("-" * 70)
    
    for stock_type in stock_types:
        symbols = loader.get_all_active_stocks(stock_type=stock_type)
        if symbols:
            print(f"\n{stock_type}:")
            for symbol in symbols[:5]:
                info = loader.load_stock_info(symbol)
                name = info.get('name', 'N/A')
                print(f"  {symbol} - {name}")
            if len(symbols) > 5:
                print(f"  ... 还有 {len(symbols) - 5} 只")
    
    print("\n" + "=" * 70)
    
    # 使用建议
    print("\n使用建议:")
    print("-" * 70)
    
    print("\n1. 只预热主板股票（推荐）:")
    print("   python scripts/prepare_training_data.py --symbols all --stock-type 主板 --workers 8 --resume")
    
    print("\n2. 只训练主板股票:")
    print("   python scripts/train_universal_model.py --stock-type 主板 --device cuda")
    
    print("\n3. 快速测试（科创板）:")
    print("   python scripts/train_universal_model.py --stock-type 科创板 --epochs 20 --device cuda")
    
    print("\n" + "=" * 70)
    print()


if __name__ == "__main__":
    main()
