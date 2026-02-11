#!/usr/bin/env python3
"""
调试批次格式
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Module.Config.config import Config
from Module.Loader.data_loader import prepare_data_loaders
import torch

def debug_batch_format():
    """调试批次格式"""
    config = Config()
    
    # 准备数据加载器
    train_loader, _, _ = prepare_data_loaders(config.data)
    
    print("检查批次格式...")
    
    # 获取第一个批次
    batch = next(iter(train_loader))
    print(f"批次类型: {type(batch)}")
    print(f"批次长度: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
    
    if isinstance(batch, (list, tuple)):
        print(f"批次是列表/元组，元素:")
        for i, item in enumerate(batch):
            print(f"  元素 {i}: 类型={type(item)}, 形状={item.shape if hasattr(item, 'shape') else 'N/A'}")
    elif isinstance(batch, dict):
        print(f"批次是字典，键: {list(batch.keys())}")
        for key, value in batch.items():
            print(f"  键 '{key}': 类型={type(value)}, 形状={value.shape if hasattr(value, 'shape') else 'N/A'}")
    
    # 检查数据集输出
    print("\n检查数据集输出...")
    sample = train_loader.dataset[0]
    print(f"样本类型: {type(sample)}")
    if isinstance(sample, (list, tuple)):
        print(f"样本是列表/元组，长度: {len(sample)}")
        for i, item in enumerate(sample):
            print(f"  元素 {i}: 类型={type(item)}, 形状={item.shape if hasattr(item, 'shape') else 'N/A'}")
    
    # 测试训练循环解包
    print("\n测试训练循环解包...")
    try:
        for batch_idx, (low_dose, full_dose) in enumerate(train_loader):
            print(f"批次 {batch_idx}: 解包成功")
            print(f"  low_dose形状: {low_dose.shape}")
            print(f"  full_dose形状: {full_dose.shape}")
            break
    except Exception as e:
        print(f"解包失败: {e}")
        # 尝试手动解包
        batch = next(iter(train_loader))
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            print("但批次是长度为2的列表/元组，可以手动解包")
            low_dose, full_dose = batch[0], batch[1]
            print(f"手动解包: low_dose形状={low_dose.shape}, full_dose形状={full_dose.shape}")

if __name__ == "__main__":
    debug_batch_format()
