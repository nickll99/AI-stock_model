"""
训练通用股票预测模型
一个模型预测所有股票

使用方法:
    # 基本训练
    python scripts/train_universal_model.py
    
    # 自定义配置
    python scripts/train_universal_model.py --model-type lstm --epochs 100 --batch-size 128
    
    # 使用GPU
    python scripts/train_universal_model.py --device cuda
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

from src.data.loader import StockDataLoader
from src.features.dataset_builder import FeatureDatasetBuilder
from src.models.universal_model import UniversalStockModel, UniversalTransformerModel


class UniversalStockDataset(Dataset):
    """通用股票数据集"""
    
    def __init__(self, X_list: List[np.ndarray], y_list: List[np.ndarray], stock_ids: List[int]):
        """
        Args:
            X_list: 特征列表，每个元素是一只股票的特征 [num_samples, seq_length, input_size]
            y_list: 标签列表，每个元素是一只股票的标签 [num_samples]
            stock_ids: 股票ID列表
        """
        self.samples = []
        
        for X, y, stock_id in zip(X_list, y_list, stock_ids):
            for i in range(len(X)):
                self.samples.append((X[i], y[i], stock_id))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        X, y, stock_id = self.samples[idx]
        return (
            torch.FloatTensor(X),
            torch.FloatTensor([y]),
            torch.LongTensor([stock_id])
        )


def load_all_stocks_data(
    symbols: List[str],
    config: Dict,
    stock_to_id: Dict[str, int],
    use_cache: bool = True,
    kline_cache_dir: str = "data/parquet",
    feature_cache_dir: str = "data/features"
) -> Tuple:
    """
    加载所有股票的数据（优先使用缓存）
    
    Args:
        symbols: 股票代码列表
        config: 配置字典
        stock_to_id: 股票ID映射
        use_cache: 是否使用缓存
        kline_cache_dir: K线缓存目录
        feature_cache_dir: 特征缓存目录
    
    Returns:
        (X_train_list, y_train_list, X_val_list, y_val_list, X_test_list, y_test_list, train_ids, val_ids, test_ids)
    """
    from src.data.cached_loader import ParquetDataLoader, FeatureCache
    
    # 使用缓存加载器
    if use_cache:
        kline_loader = ParquetDataLoader(cache_dir=kline_cache_dir)
        feature_cache = FeatureCache(cache_dir=feature_cache_dir)
        print(f"\n使用缓存数据加载 {len(symbols)} 只股票...")
        print(f"  K线缓存: {kline_cache_dir}")
        print(f"  特征缓存: {feature_cache_dir}")
    else:
        kline_loader = StockDataLoader()
        feature_cache = None
        print(f"\n从数据库加载 {len(symbols)} 只股票...")
    
    builder = FeatureDatasetBuilder()
    
    X_train_list, y_train_list, train_ids = [], [], []
    X_val_list, y_val_list, val_ids = [], [], []
    X_test_list, y_test_list, test_ids = [], [], []
    
    success_count = 0
    fail_count = 0
    cache_hit = 0
    cache_miss = 0
    
    for i, symbol in enumerate(symbols, 1):
        try:
            # 尝试从特征缓存加载（最快）
            if use_cache and feature_cache:
                try:
                    df_features = feature_cache.load_features(symbol)
                    if df_features is not None and len(df_features) >= 200:
                        cache_hit += 1
                        
                        # 直接使用缓存的特征数据
                        train_df, val_df, test_df = builder.split_dataset(
                            df_features,
                            train_ratio=0.7,
                            val_ratio=0.15,
                            test_ratio=0.15
                        )
                        
                        # 准备序列
                        X_train, y_train, feature_names = builder.prepare_sequences(
                            train_df, config['seq_length'], 'close', normalize=True
                        )
                        X_val, y_val, _ = builder.prepare_sequences(
                            val_df, config['seq_length'], 'close', feature_names, normalize=False
                        )
                        X_test, y_test, _ = builder.prepare_sequences(
                            test_df, config['seq_length'], 'close', feature_names, normalize=False
                        )
                        
                        stock_id = stock_to_id[symbol]
                        
                        X_train_list.append(X_train)
                        y_train_list.append(y_train)
                        train_ids.extend([stock_id] * len(y_train))
                        
                        X_val_list.append(X_val)
                        y_val_list.append(y_val)
                        val_ids.extend([stock_id] * len(y_val))
                        
                        X_test_list.append(X_test)
                        y_test_list.append(y_test)
                        test_ids.extend([stock_id] * len(y_test))
                        
                        success_count += 1
                        
                        if i % 100 == 0:
                            print(f"[{i}/{len(symbols)}] 已加载 {success_count} 只 (缓存命中: {cache_hit}, 未命中: {cache_miss})")
                        
                        continue
                except Exception:
                    pass  # 缓存加载失败，继续尝试其他方式
            
            # 从K线缓存或数据库加载
            cache_miss += 1
            df = kline_loader.load_kline_data(
                symbol,
                config['train_start_date'],
                config['train_end_date']
            )
            
            if len(df) < 200:
                print(f"[{i}/{len(symbols)}] {symbol}: 数据不足 ({len(df)}条)")
                fail_count += 1
                continue
            
            # 构建数据集
            dataset = builder.build_complete_dataset(
                df,
                seq_length=config['seq_length'],
                target_col='close',
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                normalize=True,
                use_cache=False
            )
            
            stock_id = stock_to_id[symbol]
            
            # 添加到列表
            X_train_list.append(dataset['X_train'])
            y_train_list.append(dataset['y_train'])
            train_ids.extend([stock_id] * len(dataset['y_train']))
            
            X_val_list.append(dataset['X_val'])
            y_val_list.append(dataset['y_val'])
            val_ids.extend([stock_id] * len(dataset['y_val']))
            
            X_test_list.append(dataset['X_test'])
            y_test_list.append(dataset['y_test'])
            test_ids.extend([stock_id] * len(dataset['y_test']))
            
            success_count += 1
            
            if i % 100 == 0:
                print(f"[{i}/{len(symbols)}] 已加载 {success_count} 只股票")
            
        except Exception as e:
            print(f"[{i}/{len(symbols)}] {symbol}: 失败 - {e}")
            fail_count += 1
    
    print(f"\n数据加载完成:")
    print(f"  成功: {success_count} 只")
    print(f"  失败: {fail_count} 只")
    if use_cache:
        print(f"  缓存命中: {cache_hit} 只 ({cache_hit/(cache_hit+cache_miss)*100:.1f}%)")
        print(f"  缓存未命中: {cache_miss} 只")
    print(f"  训练样本: {sum(len(y) for y in y_train_list):,}")
    print(f"  验证样本: {sum(len(y) for y in y_val_list):,}")
    print(f"  测试样本: {sum(len(y) for y in y_test_list):,}")
    
    return (X_train_list, y_train_list, X_val_list, y_val_list, 
            X_test_list, y_test_list, train_ids, val_ids, test_ids)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch, stock_ids_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        stock_ids_batch = stock_ids_batch.squeeze().to(device)
        
        # 前向传播
        outputs = model(X_batch, stock_ids_batch)
        loss = criterion(outputs, y_batch)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch, stock_ids_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            stock_ids_batch = stock_ids_batch.squeeze().to(device)
            
            outputs = model(X_batch, stock_ids_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练通用股票预测模型")
    
    parser.add_argument("--model-type", type=str, default="lstm", choices=["lstm", "gru", "transformer"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--stock-embedding-dim", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="限制股票数量（测试用）")
    parser.add_argument("--output-dir", type=str, default="out/universal_model")
    parser.add_argument("--no-cache", action="store_true", help="不使用缓存，从数据库加载")
    parser.add_argument("--kline-cache-dir", type=str, default="data/parquet", help="K线缓存目录")
    parser.add_argument("--feature-cache-dir", type=str, default="data/features", help="特征缓存目录")
    parser.add_argument("--stock-type", type=str, default=None, help="股票类型筛选（如'主板'、'创业板'、'科创板'等），默认不筛选")
    
    args = parser.parse_args()
    
    # 处理缓存参数（默认使用缓存，除非指定 --no-cache）
    use_cache = not args.no_cache
    
    print("\n" + "=" * 70)
    print("  训练通用股票预测模型")
    print("=" * 70)
    
    # 配置
    config = {
        "model_type": args.model_type,
        "seq_length": 60,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "stock_embedding_dim": args.stock_embedding_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_start_date": "2021-01-01",
        "train_end_date": "2024-12-31"
    }
    
    print(f"\n配置:")
    print(f"  模型类型: {config['model_type']}")
    print(f"  序列长度: {config['seq_length']}")
    print(f"  隐藏层大小: {config['hidden_size']}")
    print(f"  股票嵌入维度: {config['stock_embedding_dim']}")
    print(f"  训练轮数: {config['epochs']}")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  设备: {args.device}")
    print(f"  数据源: {'缓存数据' if use_cache else 'MySQL数据库'}")
    if use_cache:
        print(f"    K线缓存: {args.kline_cache_dir}")
        print(f"    特征缓存: {args.feature_cache_dir}")
    
    # 获取股票列表
    print("\n获取股票列表...")
    loader = StockDataLoader()
    symbols = loader.get_all_active_stocks(stock_type=args.stock_type)
    
    if args.stock_type:
        print(f"股票类型: {args.stock_type}")
    
    if args.limit:
        symbols = symbols[:args.limit]
    
    print(f"股票数量: {len(symbols)}")
    
    # 创建股票ID映射
    stock_to_id = {symbol: i for i, symbol in enumerate(symbols)}
    id_to_stock = {i: symbol for symbol, i in stock_to_id.items()}
    
    # 加载数据
    (X_train_list, y_train_list, X_val_list, y_val_list,
     X_test_list, y_test_list, train_ids, val_ids, test_ids) = load_all_stocks_data(
        symbols, config, stock_to_id,
        use_cache=use_cache,
        kline_cache_dir=args.kline_cache_dir,
        feature_cache_dir=args.feature_cache_dir
    )
    
    # 创建数据集
    print("\n创建数据集...")
    train_dataset = UniversalStockDataset(X_train_list, y_train_list, train_ids)
    val_dataset = UniversalStockDataset(X_val_list, y_val_list, val_ids)
    test_dataset = UniversalStockDataset(X_test_list, y_test_list, test_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    print(f"✓ 训练集: {len(train_dataset):,} 样本")
    print(f"✓ 验证集: {len(val_dataset):,} 样本")
    print(f"✓ 测试集: {len(test_dataset):,} 样本")
    
    # 创建模型
    print("\n创建模型...")
    input_size = X_train_list[0].shape[2]
    
    if config['model_type'] == 'transformer':
        model = UniversalTransformerModel(
            num_stocks=len(symbols),
            input_size=input_size,
            d_model=config['hidden_size'],
            nhead=4,
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            stock_embedding_dim=config['stock_embedding_dim']
        )
    else:
        model = UniversalStockModel(
            num_stocks=len(symbols),
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            stock_embedding_dim=config['stock_embedding_dim'],
            model_type=config['model_type']
        )
    
    model = model.to(args.device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型参数量: {total_params:,}")
    
    # 优化器和损失函数
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 训练
    print(f"\n开始训练...")
    print(f"{'='*70}\n")
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, args.device)
        
        # 验证
        val_loss = evaluate(model, val_loader, criterion, args.device)
        
        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch [{epoch+1}/{config['epochs']}] - "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"LR: {current_lr:.6f}, "
              f"Time: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'stock_to_id': stock_to_id,
                'id_to_stock': id_to_stock
            }, output_dir / 'best_model.pth')
            
            print(f"  ✓ 最佳模型已保存")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            print(f"\n早停触发，在第 {epoch+1} 轮停止训练")
            break
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"训练完成！")
    print(f"  总耗时: {total_time/60:.1f} 分钟")
    print(f"  最佳验证损失: {best_val_loss:.6f}")
    print(f"{'='*70}")
    
    # 保存训练历史
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n输出目录: {output_dir}/")
    print(f"  - best_model.pth (最佳模型)")
    print(f"  - training_history.json (训练历史)")
    print(f"  - config.json (配置文件)")
    print()


if __name__ == "__main__":
    main()
