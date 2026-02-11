#!/usr/bin/env python3
"""
调试MONAI变换的输入输出格式
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Module.Config.config import Config
from Module.Loader.data_loader import get_transforms
import numpy as np

def test_monai_transform():
    """测试MONAI变换的输入输出"""
    config = Config()
    
    print("测试MONAI变换...")
    
    # 创建虚拟输入（模拟CTDataset的输出）
    # 注意：_load_image返回的形状是 (C, H, W, D) 其中 C=1
    dummy_low = np.random.randn(1, 64, 64, 32).astype(np.float32)
    dummy_full = np.random.randn(1, 64, 64, 32).astype(np.float32)
    
    print(f"输入 low 形状: {dummy_low.shape}")
    print(f"输入 full 形状: {dummy_full.shape}")
    
    # 获取变换
    transform = get_transforms(config.data, is_train=True)
    print(f"\n变换管道: {transform}")
    
    # 测试1：直接应用变换到字典
    print("\n=== 测试1：应用变换到字典 ===")
    input_dict = {"low": dummy_low, "full": dummy_full}
    print(f"输入字典类型: {type(input_dict)}")
    print(f"输入字典键: {list(input_dict.keys())}")
    
    try:
        output = transform(input_dict)
        print(f"输出类型: {type(output)}")
        if isinstance(output, dict):
            print(f"输出字典键: {list(output.keys())}")
            for key, value in output.items():
                print(f"  键 '{key}': 类型={type(value)}, 形状={value.shape if hasattr(value, 'shape') else 'N/A'}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2：检查EnsureChannelFirst变换
    print("\n=== 测试2：单独测试EnsureChannelFirst ===")
    from monai.transforms import EnsureChannelFirst
    
    ensure_channel = EnsureChannelFirst(channel_dim=0)
    print(f"EnsureChannelFirst变换: {ensure_channel}")
    
    # 测试单个数组
    try:
        single_output = ensure_channel(dummy_low)
        print(f"单个数组输出: 类型={type(single_output)}, 形状={single_output.shape}")
    except Exception as e:
        print(f"单个数组错误: {e}")
    
    # 测试字典
    try:
        dict_output = ensure_channel(input_dict)
        print(f"字典输出: 类型={type(dict_output)}")
    except Exception as e:
        print(f"字典错误: {e}")
    
    # 测试3：检查MONAI文档建议的方式
    print("\n=== 测试3：使用MONAI推荐的方式 ===")
    from monai.transforms import Compose
    
    # 创建简单的变换管道
    simple_transforms = Compose([
        EnsureChannelFirst(channel_dim=0, keys=["low", "full"]),
    ])
    
    print(f"带keys参数的变换: {simple_transforms}")
    
    try:
        simple_output = simple_transforms(input_dict)
        print(f"输出类型: {type(simple_output)}")
        if isinstance(simple_output, dict):
            print(f"输出键: {list(simple_output.keys())}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试4：检查当前代码中的变换
    print("\n=== 测试4：检查当前代码中的问题 ===")
    print("当前get_transforms返回的变换:")
    for i, t in enumerate(transform.transforms):
        print(f"  {i}: {t}")
        if hasattr(t, 'keys'):
            print(f"      keys参数: {getattr(t, 'keys', '无')}")

if __name__ == "__main__":
    test_monai_transform()
