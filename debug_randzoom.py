#!/usr/bin/env python3
"""
调试RandZoomd问题
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from monai.transforms import RandZoomD

def test_randzoom():
    """测试RandZoomd变换"""
    print("测试RandZoomd变换...")
    
    # 创建测试数据（深度维度为1）
    test_data = {
        "low": np.random.randn(1, 256, 256, 1).astype(np.float32),
        "full": np.random.randn(1, 256, 256, 1).astype(np.float32)
    }
    
    print(f"原始形状: low={test_data['low'].shape}, full={test_data['full'].shape}")
    
    # 测试不同的参数
    test_cases = [
        {"min_zoom": 0.9, "max_zoom": 1.1, "prob": 1.0},
        {"min_zoom": 0.95, "max_zoom": 1.05, "prob": 1.0},
        {"min_zoom": 1.0, "max_zoom": 1.0, "prob": 1.0},  # 无缩放
    ]
    
    for i, params in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: {params}")
        transform = RandZoomD(keys=["low", "full"], **params)
        
        try:
            result = transform(test_data.copy())
            print(f"  成功！low形状={result['low'].shape}, full形状={result['full'].shape}")
            
            # 检查深度维度
            if result['low'].shape[-1] == 0:
                print(f"  警告：深度维度为0！")
        except Exception as e:
            print(f"  错误: {e}")
    
    # 测试多次运行以查看随机性
    print("\n测试随机性（运行10次）:")
    transform = RandZoomD(keys=["low", "full"], min_zoom=0.9, max_zoom=1.1, prob=1.0)
    
    zero_depth_count = 0
    for i in range(10):
        try:
            result = transform(test_data.copy())
            depth = result['low'].shape[-1]
            print(f"  运行 {i+1}: 深度维度={depth}")
            if depth == 0:
                zero_depth_count += 1
        except Exception as e:
            print(f"  运行 {i+1}: 错误 - {e}")
    
    print(f"\n深度维度为0的次数: {zero_depth_count}/10")
    
    # 检查MONAI内部实现
    print("\n检查缩放计算:")
    from monai.transforms.spatial.array import RandZoom
    
    # 手动计算缩放
    zoom_transform = RandZoom(min_zoom=0.9, max_zoom=1.1, prob=1.0)
    
    # 创建测试张量
    test_tensor = torch.randn(1, 256, 256, 1)
    print(f"测试张量形状: {test_tensor.shape}")
    
    # 模拟缩放计算
    import math
    zoom_factor = 0.9  # 最坏情况
    new_depth = math.floor(1 * zoom_factor)
    print(f"缩放因子 {zoom_factor}: 1 * {zoom_factor} = {new_depth} (floor)")
    
    zoom_factor = 1.1  # 最好情况
    new_depth = math.floor(1 * zoom_factor)
    print(f"缩放因子 {zoom_factor}: 1 * {zoom_factor} = {new_depth} (floor)")

if __name__ == "__main__":
    test_randzoom()
