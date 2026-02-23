"""
图像质量指标计算模块

提供ImageMetricsCalculator类，用于计算各种图像质量指标：
- RMSE（均方根误差）
- MAE（平均绝对误差）
- PSNR（峰值信噪比）
- SSIM（结构相似性）
- MS-SSIM（多尺度SSIM）
- LPIPS（学习感知图像块相似度）
"""

import os
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

# 尝试导入scipy.signal中的convolve2d
try:
    from scipy.signal import convolve2d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # 如果scipy不可用，定义一个简单的convolve2d实现
    def convolve2d(image, kernel, mode='valid'):
        """
        简单的2D卷积实现（仅支持mode='valid'）
        
        参数:
            image: 输入图像 (2D数组)
            kernel: 卷积核 (2D数组)
            mode: 卷积模式，仅支持'valid'
            
        返回:
            卷积结果
        """
        if mode != 'valid':
            raise NotImplementedError("简化convolve2d仅支持mode='valid'")
        
        kh, kw = kernel.shape
        ih, iw = image.shape
        
        # 计算输出尺寸
        oh = ih - kh + 1
        ow = iw - kw + 1
        
        # 执行卷积
        result = np.zeros((oh, ow))
        for i in range(oh):
            for j in range(ow):
                result[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
        
        return result

# 尝试导入skimage指标
try:
    from skimage.metrics import (
        peak_signal_noise_ratio,
        structural_similarity,
        mean_squared_error
    )
    SKIMAGE_METRICS_AVAILABLE = True
except ImportError:
    SKIMAGE_METRICS_AVAILABLE = False
    warnings.warn("skimage.metrics不可用，使用numpy实现")

# 尝试导入LPIPS
try:
    import lpips as lpips_lib
    lpips_loss = lpips_lib.LPIPS(net='alex', verbose=False)
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    warnings.warn("LPIPS不可用，请安装lpips包：pip install lpips")
except Exception as e:
    LPIPS_AVAILABLE = False
    warnings.warn(f"LPIPS初始化失败: {e}")

# 导入配置
from ..config import DiagnosticsConfig

# 定义mean_absolute_error函数（skimage.metrics中没有此函数）
def mean_absolute_error(target: np.ndarray, pred: np.ndarray) -> float:
    """
    计算平均绝对误差 (MAE)
    
    参数:
        target: 目标图像数组
        pred: 预测图像数组
        
    返回:
        平均绝对误差值
    """
    return np.mean(np.abs(target - pred))

# 如果skimage指标不可用，定义其他替代函数
if not SKIMAGE_METRICS_AVAILABLE:
    def mean_squared_error(target: np.ndarray, pred: np.ndarray) -> float:
        """
        计算均方误差 (MSE)
        
        参数:
            target: 目标图像数组
            pred: 预测图像数组
            
        返回:
            均方误差值
        """
        return np.mean((target - pred) ** 2)
    
    def peak_signal_noise_ratio(
        image_true: np.ndarray,
        image_test: np.ndarray,
        *,
        data_range: Optional[float] = None
    ) -> float:
        """
        计算峰值信噪比 (PSNR)
        
        参数:
            image_true: 原始图像数组
            image_test: 测试图像数组
            data_range: 图像数据范围，如果为None则自动计算
            
        返回:
            PSNR值 (dB)
        """
        # 确保输入为numpy数组
        image_true = np.asarray(image_true)
        image_test = np.asarray(image_test)
        
        # 自动计算数据范围
        if data_range is None:
            if image_true.dtype.kind in 'iu':
                # 整数类型
                bit_depth = np.iinfo(image_true.dtype).bits
                data_range = 2 ** bit_depth - 1
            else:
                # 浮点类型，假设范围为[0, 1]
                data_range = 1.0
        
        # 计算均方误差
        mse = mean_squared_error(image_true, image_test)
        
        # 避免除以零
        if mse == 0:
            return float('inf')
        
        # 计算PSNR
        psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
        return float(psnr)
    
    def structural_similarity(
        im1: np.ndarray,
        im2: np.ndarray,
        *,
        win_size: Optional[int] = None,
        gradient: bool = False,
        data_range: Optional[float] = None,
        channel_axis: Optional[int] = None,
        gaussian_weights: bool = False,
        full: bool = False,
        **kwargs
    ) -> Union[float, np.ndarray]:
        """
        计算结构相似性指数 (SSIM)
        
        注意: 这是一个简化实现，仅支持2D灰度图像。
        对于彩色图像和多通道图像，建议安装skimage以获得完整功能。
        
        参数:
            im1: 第一幅图像
            im2: 第二幅图像
            win_size: 滑动窗口大小，必须为奇数
            gradient: 是否计算梯度 (不支持)
            data_range: 图像数据范围
            channel_axis: 通道轴 (不支持)
            gaussian_weights: 是否使用高斯权重 (不支持)
            full: 是否返回完整SSIM图 (不支持)
            **kwargs: 其他参数 (忽略)
            
        返回:
            SSIM值 (如果full=False) 或 SSIM图 (如果full=True)
        """
        # 简化实现：仅支持2D灰度图像
        im1 = np.asarray(im1)
        im2 = np.asarray(im2)
        
        # 检查维度
        if im1.ndim != 2 or im2.ndim != 2:
            raise NotImplementedError(
                "简化SSIM实现仅支持2D灰度图像。请安装scikit-image以获得完整功能。"
            )
        
        # 设置默认窗口大小
        if win_size is None:
            win_size = 7
        
        # 确保窗口大小为奇数
        if win_size % 2 == 0:
            win_size += 1
        
        # 确保窗口大小不超过图像尺寸
        min_dim = min(im1.shape[0], im1.shape[1])
        if win_size > min_dim:
            win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
        
        if win_size < 3:
            return 0.0
        
        # 计算数据范围
        if data_range is None:
            data_range = 1.0
        
        # 计算局部均值和方差
        pad = win_size // 2
        im1_pad = np.pad(im1, pad, mode='reflect')
        im2_pad = np.pad(im2, pad, mode='reflect')
        
        # 使用均匀滤波器计算局部均值
        kernel = np.ones((win_size, win_size)) / (win_size * win_size)
        mu1 = convolve2d(im1_pad, kernel, mode='valid')
        mu2 = convolve2d(im2_pad, kernel, mode='valid')
        
        # 计算局部方差和协方差
        sigma1_sq = convolve2d(im1_pad**2, kernel, mode='valid') - mu1**2
        sigma2_sq = convolve2d(im2_pad**2, kernel, mode='valid') - mu2**2
        sigma12 = convolve2d(im1_pad * im2_pad, kernel, mode='valid') - mu1 * mu2
        
        # 稳定性常数
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        # 计算SSIM图
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if full:
            return ssim_map
        else:
            return float(ssim_map.mean())

# 导入ImageMetricsCalculator
from .calculator import ImageMetricsCalculator

__all__ = ['ImageMetricsCalculator', 'SKIMAGE_METRICS_AVAILABLE', 'LPIPS_AVAILABLE',
           'mean_squared_error', 'mean_absolute_error', 'peak_signal_noise_ratio',
           'structural_similarity', 'lpips_loss', 'DiagnosticsConfig']
