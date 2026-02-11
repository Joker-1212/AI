#!/usr/bin/env python3
"""
调试数据形状问题
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Module.Config.config import Config
from Module.Loader.data_loader import CTDataset, get_transforms, create_dummy_data
import numpy as np

def debug_data_shapes():
    """调试数据形状"""
    config = Config()
    
    print("配置信息:")
    print(f"  image_size: {config.data.image_size}")
    print(f"  normalize: {config.data.normalize}")
    print(f"  normalize_range: {config.data.normalize_range}")
    
    # 创建虚拟数据
    print("\n创建虚拟数据...")
    create_dummy_data(config.data, num_samples=2)
    
    # 查找文件
    import glob
    low_dose_pattern = os.path.join(config.data.data_dir, config.data.low_dose_dir, "*")
    full_dose_pattern = os.path.join(config.data.data_dir, config.data.full_dose_dir, "*")
    
    low_dose_files = sorted(glob.glob(low_dose_pattern))
    full_dose_files = sorted(glob.glob(full_dose_pattern))
    
    print(f"\n找到 {len(low_dose_files)} 个低剂量文件")
    print(f"找到 {len(full_dose_files)} 个全剂量文件")
    
    # 创建数据集
    dataset = CTDataset(
        low_dose_files[:2],
        full_dose_files[:2],
        config.data,
        transform=get_transforms(config.data, is_train=True),
        is_train=True
    )
    
    print("\n检查原始数据形状（无变换）:")
    raw_dataset = CTDataset(
        low_dose_files[:2],
        full_dose_files[:2],
        config.data,
        transform=None,
        is_train=True
    )
    
    for i in range(2):
        low, full = raw_dataset[i]
        print(f"样本 {i}:")
        print(f"  low形状: {low.shape}, dtype: {low.dtype}")
        print(f"  full形状: {full.shape}, dtype: {full.dtype}")
    
    print("\n检查变换后的数据形状:")
    for i in range(2):
        try:
            low, full = dataset[i]
            print(f"样本 {i}:")
            print(f"  low形状: {low.shape}, dtype: {low.dtype}")
            print(f"  full形状: {full.shape}, dtype: {full.dtype}")
        except Exception as e:
            print(f"样本 {i} 错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 检查变换管道
    print("\n检查变换管道...")
    transform = get_transforms(config.data, is_train=True)
    print(f"变换数量: {len(transform.transforms)}")
    
    # 逐步应用变换
    print("\n逐步应用变换:")
    from monai.transforms import Compose
    
    # 创建测试数据
    test_low = np.random.randn(1, 256, 256, 1).astype(np.float32)
    test_full = np.random.randn(1, 256, 256, 1).astype(np.float32)
    test_dict = {"low": test_low, "full": test_full}
    
    print(f"初始形状: low={test_low.shape}, full={test_full.shape}")
    
    current_data = test_dict
    for i, t in enumerate(transform.transforms):
        print(f"\n变换 {i}: {t}")
        try:
            current_data = t(current_data)
            if isinstance(current_data, dict):
                for key in ["low", "full"]:
                    if key in current_data:
                        val = current_data[key]
                        print(f"  {key}: 类型={type(val).__name__}, 形状={getattr(val, 'shape', 'N/A')}")
        except Exception as e:
            print(f"  错误: {e}")
            break

if __name__ == "__main__":
    debug_data_shapes()
