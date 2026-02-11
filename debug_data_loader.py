#!/usr/bin/env python3
"""
调试脚本：检查数据加载器的输出格式
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Module.Config.config import Config
from Module.Loader.data_loader import prepare_data_loaders, create_dummy_data
import torch

def test_data_loader_output():
    """测试数据加载器的输出格式"""
    # 加载配置
    config = Config()
    
    # 创建虚拟数据（如果不存在）
    data_dir = config.data.data_dir
    low_dir = os.path.join(data_dir, config.data.low_dose_dir)
    full_dir = os.path.join(data_dir, config.data.full_dose_dir)
    
    if not os.path.exists(low_dir) or not os.listdir(low_dir):
        print("创建虚拟数据...")
        create_dummy_data(config.data, num_samples=5)
    
    # 准备数据加载器
    print("准备数据加载器...")
    train_loader, val_loader, test_loader = prepare_data_loaders(config.data)
    
    # 检查训练加载器的第一个批次
    print("\n检查训练加载器输出...")
    for i, batch in enumerate(train_loader):
        print(f"\n批次 {i}:")
        print(f"  类型: {type(batch)}")
        
        if isinstance(batch, tuple):
            print(f"  元组长度: {len(batch)}")
            for j, item in enumerate(batch):
                print(f"    元素 {j}: 类型={type(item)}, 形状={item.shape if hasattr(item, 'shape') else 'N/A'}")
                if torch.is_tensor(item):
                    print(f"          数据类型={item.dtype}, 设备={item.device}")
        elif isinstance(batch, dict):
            print(f"  字典键: {list(batch.keys())}")
            for key, value in batch.items():
                print(f"    键 '{key}': 类型={type(value)}, 形状={value.shape if hasattr(value, 'shape') else 'N/A'}")
                if torch.is_tensor(value):
                    print(f"          数据类型={value.dtype}, 设备={value.device}")
        else:
            print(f"  其他类型: {type(batch)}")
        
        # 只检查前2个批次
        if i >= 1:
            break
    
    # 检查单个样本
    print("\n检查单个样本（从数据集中获取）...")
    dataset = train_loader.dataset
    sample = dataset[0]
    print(f"单个样本类型: {type(sample)}")
    if isinstance(sample, tuple):
        print(f"  元组长度: {len(sample)}")
        for j, item in enumerate(sample):
            print(f"    元素 {j}: 类型={type(item)}, 形状={item.shape if hasattr(item, 'shape') else 'N/A'}")
    elif isinstance(sample, dict):
        print(f"  字典键: {list(sample.keys())}")
    
    # 检查MONAI变换的输出
    print("\n检查MONAI变换...")
    from Module.Loader.data_loader import get_transforms
    import numpy as np
    
    # 创建虚拟输入
    dummy_low = np.random.randn(1, 64, 64, 32).astype(np.float32)
    dummy_full = np.random.randn(1, 64, 64, 32).astype(np.float32)
    
    transform = get_transforms(config.data, is_train=True)
    print(f"变换类型: {type(transform)}")
    
    # 测试变换
    input_dict = {"low": dummy_low, "full": dummy_full}
    output = transform(input_dict)
    print(f"变换输出类型: {type(output)}")
    if isinstance(output, dict):
        print(f"  输出键: {list(output.keys())}")
        for key, value in output.items():
            print(f"    键 '{key}': 类型={type(value)}, 形状={value.shape if hasattr(value, 'shape') else 'N/A'}")

if __name__ == "__main__":
    test_data_loader_output()
