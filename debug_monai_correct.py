#!/usr/bin/env python3
"""
调试MONAI变换的正确用法
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

def test_monai_dict_transform():
    """测试MONAI字典变换的正确用法"""
    print("测试MONAI字典变换...")
    
    # 创建虚拟输入
    dummy_low = np.random.randn(1, 64, 64, 32).astype(np.float32)
    dummy_full = np.random.randn(1, 64, 64, 32).astype(np.float32)
    
    print(f"输入 low 形状: {dummy_low.shape}")
    print(f"输入 full 形状: {dummy_full.shape}")
    
    # 方法1：使用MapTransform子类（MONAI推荐的方式）
    print("\n=== 方法1：使用MapTransform子类 ===")
    from monai.transforms import (
        Compose,
        EnsureChannelFirstD,  # 字典版本的EnsureChannelFirst
        ScaleIntensityRangeD,
        RandRotateD,
        ToTensorD
    )
    
    # 创建字典变换管道
    dict_transforms = Compose([
        EnsureChannelFirstD(keys=["low", "full"], channel_dim=0),
        ScaleIntensityRangeD(
            keys=["low", "full"],
            a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True
        ),
        ToTensorD(keys=["low", "full"], dtype=torch.float32),
    ])
    
    input_dict = {"low": dummy_low, "full": dummy_full}
    
    try:
        output = dict_transforms(input_dict)
        print(f"成功！输出类型: {type(output)}")
        print(f"输出键: {list(output.keys())}")
        for key, value in output.items():
            print(f"  键 '{key}': 类型={type(value)}, 形状={value.shape}, dtype={value.dtype}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 方法2：检查当前代码中的变换
    print("\n=== 方法2：检查当前代码中的变换 ===")
    from Module.Config.config import Config
    from Module.Loader.data_loader import get_transforms
    
    config = Config()
    current_transform = get_transforms(config.data, is_train=True)
    
    print(f"当前变换类型: {type(current_transform)}")
    print(f"变换数量: {len(current_transform.transforms)}")
    
    # 检查每个变换
    for i, t in enumerate(current_transform.transforms):
        print(f"\n变换 {i}: {t}")
        print(f"  类型: {type(t)}")
        
        # 检查是否有keys属性
        if hasattr(t, 'keys'):
            print(f"  keys: {t.keys}")
        else:
            print(f"  无keys属性 - 这可能是个问题！")
            
        # 检查是否是MapTransform
        from monai.transforms import MapTransform
        if isinstance(t, MapTransform):
            print(f"  是MapTransform子类")
        else:
            print(f"  不是MapTransform子类")
    
    # 方法3：测试修复方案
    print("\n=== 方法3：测试修复方案 ===")
    
    # 创建修复后的变换管道
    from monai.transforms import (
        EnsureChannelFirst,
        ScaleIntensityRange,
        Resize,
        RandRotate,
        RandFlip,
        RandZoom,
        RandGaussianNoise,
        RandAdjustContrast,
        ToTensor
    )
    
    # 注意：我们需要使用字典版本的变换
    fixed_transforms = Compose([
        EnsureChannelFirstD(keys=["low", "full"], channel_dim=0),
        ScaleIntensityRangeD(
            keys=["low", "full"],
            a_min=config.data.normalize_range[0],
            a_max=config.data.normalize_range[1],
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        # Resize没有字典版本，需要自定义
        # 暂时跳过Resize进行测试
        ToTensorD(keys=["low", "full"], dtype=torch.float32),
    ])
    
    try:
        fixed_output = fixed_transforms(input_dict)
        print(f"修复方案成功！")
        print(f"输出类型: {type(fixed_output)}")
    except Exception as e:
        print(f"修复方案错误: {e}")

if __name__ == "__main__":
    test_monai_dict_transform()
