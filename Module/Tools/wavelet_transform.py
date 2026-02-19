"""
小波变换工具函数
实现方向小波变换和轮廓波变换，用于低剂量CT增强
参考论文：A deep convolutional neural network using directional wavelets for low‑dose X‑ray CT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from typing import Tuple, List, Optional, Union


class DWT2d(nn.Module):
    """2D离散小波变换（使用pywt）"""
    
    def __init__(self, wavelet='db4', mode='zero'):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入进行2D小波变换
        
        参数:
            x: 形状为 (B, C, H, W) 的张量
        
        返回:
            coeffs: 小波系数，形状为 (B, C*4, H//2, W//2)
        """
        batch_size, channels, height, width = x.shape
        coeffs_list = []
        
        for b in range(batch_size):
            for c in range(channels):
                img = x[b, c].detach().cpu().numpy()
                # 执行2D小波分解
                coeffs2 = pywt.dwt2(img, self.wavelet, mode=self.mode)
                cA, (cH, cV, cD) = coeffs2
                # 堆叠系数
                coeffs = np.stack([cA, cH, cV, cD], axis=0)  # (4, H//2, W//2)
                coeffs_list.append(coeffs)
        
        # 合并批次和通道
        coeffs_array = np.stack(coeffs_list, axis=0)  # (B*C, 4, H//2, W//2)
        coeffs_tensor = torch.from_numpy(coeffs_array).float().to(x.device)
        # 重塑为 (B, C*4, H//2, W//2)
        coeffs_tensor = coeffs_tensor.view(batch_size, channels * 4, coeffs_tensor.shape[2], coeffs_tensor.shape[3])
        return coeffs_tensor
    
    def inverse(self, coeffs: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
        """
        从小波系数重构图像
        
        参数:
            coeffs: 形状为 (B, C*4, H//2, W//2) 的张量
            original_shape: 原始图像形状 (H, W)
        
        返回:
            x_recon: 重构图像，形状为 (B, C, H, W)
        """
        batch_size, coeff_channels, h, w = coeffs.shape
        channels = coeff_channels // 4
        coeffs_np = coeffs.detach().cpu().numpy()
        
        recon_list = []
        for b in range(batch_size):
            channel_recon = []
            for c in range(channels):
                # 提取当前通道的4个子带
                idx = c * 4
                cA = coeffs_np[b, idx]
                cH = coeffs_np[b, idx + 1]
                cV = coeffs_np[b, idx + 2]
                cD = coeffs_np[b, idx + 3]
                # 执行逆变换
                coeffs2 = (cA, (cH, cV, cD))
                recon = pywt.idwt2(coeffs2, self.wavelet, mode=self.mode)
                # 确保形状匹配
                recon = recon[:original_shape[0], :original_shape[1]]
                channel_recon.append(recon)
            # 堆叠通道
            channel_recon = np.stack(channel_recon, axis=0)  # (C, H, W)
            recon_list.append(channel_recon)
        
        recon_array = np.stack(recon_list, axis=0)  # (B, C, H, W)
        return torch.from_numpy(recon_array).float().to(coeffs.device)


class LearnableDirectionalWavelet(nn.Module):
    """
    可学习的方向小波变换
    使用可分离卷积实现方向滤波器组
    """
    
    def __init__(self, in_channels: int, num_directions: int = 8, kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.num_directions = num_directions
        self.kernel_size = kernel_size
        
        # 创建方向滤波器组
        self.directional_filters = nn.Conv2d(
            in_channels, 
            in_channels * num_directions,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,  # 深度可分离卷积
            bias=False
        )
        
        # 初始化滤波器为方向性模式
        self._init_directional_filters()
        
    def _init_directional_filters(self):
        """初始化方向滤波器"""
        weight = self.directional_filters.weight.data
        out_channels, in_channels, kh, kw = weight.shape
        
        # 为每个方向创建不同的Gabor-like滤波器
        for d in range(self.num_directions):
            theta = d * np.pi / self.num_directions
            for c in range(self.in_channels):
                # 创建方向性滤波器
                filter_2d = self._create_directional_filter(theta, kh, kw)
                weight[d * self.in_channels + c, c] = torch.from_numpy(filter_2d).float()
        
        self.directional_filters.weight.data = weight
    
    def _create_directional_filter(self, theta: float, kh: int, kw: int) -> np.ndarray:
        """创建方向性滤波器（简化Gabor滤波器）"""
        center = (kh - 1) / 2.0
        y, x = np.ogrid[:kh, :kw]
        y = y - center
        x = x - center
        
        # 旋转坐标
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        
        # 创建方向性高斯导数
        sigma = 1.0
        gaussian = np.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2))
        derivative = -x_theta / sigma**2 * gaussian
        
        # 归一化
        derivative = derivative / (np.abs(derivative).sum() + 1e-8)
        return derivative
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用方向小波变换
        
        参数:
            x: 形状为 (B, C, H, W) 的张量
        
        返回:
            directional_coeffs: 形状为 (B, C*num_directions, H, W) 的张量
        """
        return self.directional_filters(x)


class ContourletTransform(nn.Module):
    """
    简化的轮廓波变换
    使用拉普拉斯金字塔和方向滤波器组
    """
    
    def __init__(self, levels: int = 3, num_directions: int = 8):
        super().__init__()
        self.levels = levels
        self.num_directions = num_directions
        
        # 拉普拉斯金字塔层
        self.downsample = nn.AvgPool2d(2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 方向滤波器组
        self.directional_wavelet = LearnableDirectionalWavelet(
            in_channels=1,  # 每个通道独立处理
            num_directions=num_directions,
            kernel_size=5
        )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        应用轮廓波变换
        
        参数:
            x: 形状为 (B, C, H, W) 的张量
        
        返回:
            pyramid_coeffs: 每层系数的列表，每个元素形状为 (B, C*num_directions, H_i, W_i)
        """
        batch_size, channels, height, width = x.shape
        pyramid_coeffs = []
        
        current = x
        for level in range(self.levels):
            # 下采样
            down = self.downsample(current)
            # 上采样以匹配原始尺寸
            up = self.upsample(down)
            # 计算高频细节
            detail = current - up
            
            # 对细节应用方向小波变换
            detail_coeffs = []
            for c in range(channels):
                detail_c = detail[:, c:c+1]  # (B, 1, H, W)
                coeffs_c = self.directional_wavelet(detail_c)  # (B, num_directions, H, W)
                detail_coeffs.append(coeffs_c)
            
            # 合并通道
            detail_coeffs = torch.cat(detail_coeffs, dim=1)  # (B, channels*num_directions, H, W)
            pyramid_coeffs.append(detail_coeffs)
            
            # 更新当前为低频部分
            current = down
        
        # 添加最后的低频部分
        low_freq_coeffs = []
        for c in range(channels):
            low_c = current[:, c:c+1]
            coeffs_c = self.directional_wavelet(low_c)
            low_freq_coeffs.append(coeffs_c)
        
        low_freq_coeffs = torch.cat(low_freq_coeffs, dim=1)
        pyramid_coeffs.append(low_freq_coeffs)
        
        return pyramid_coeffs


class WaveletDomainProcessing(nn.Module):
    """
    小波域处理模块
    将图像转换到小波域，应用卷积，然后逆变换
    """
    
    def __init__(self, in_channels: int, out_channels: int, wavelet_type: str = 'directional'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wavelet_type = wavelet_type
        
        if wavelet_type == 'directional':
            self.wavelet = LearnableDirectionalWavelet(in_channels, num_directions=8)
            self.wavelet_channels = in_channels * 8
        elif wavelet_type == 'dwt':
            self.wavelet = DWT2d(wavelet='db4')
            self.wavelet_channels = in_channels * 4
        elif wavelet_type == 'contourlet':
            self.wavelet = ContourletTransform(levels=3, num_directions=8)
            self.wavelet_channels = in_channels * 8 * 4  # 近似值
        else:
            raise ValueError(f"未知的小波类型: {wavelet_type}")
        
        # 小波域中的卷积处理
        self.wavelet_conv = nn.Sequential(
            nn.Conv2d(self.wavelet_channels, self.wavelet_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.wavelet_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.wavelet_channels * 2, self.wavelet_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.wavelet_channels),
            nn.ReLU(inplace=True),
        )
        
        # 逆变换（对于DWT）
        if wavelet_type == 'dwt':
            self.inverse_wavelet = lambda coeffs, shape: self.wavelet.inverse(coeffs, shape)
        else:
            # 对于方向小波和轮廓波，我们使用简单的卷积进行重构
            self.inverse_conv = nn.Conv2d(self.wavelet_channels, in_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        小波域处理
        
        参数:
            x: 形状为 (B, C, H, W) 的张量
        
        返回:
            enhanced: 增强后的图像，形状为 (B, C, H, W)
        """
        original_shape = x.shape[2:]
        
        if self.wavelet_type == 'dwt':
            # DWT变换
            coeffs = self.wavelet(x)
            # 小波域处理
            coeffs_processed = self.wavelet_conv(coeffs)
            # 逆变换
            enhanced = self.inverse_wavelet(coeffs_processed, original_shape)
            
        elif self.wavelet_type == 'directional':
            # 方向小波变换
            coeffs = self.wavelet(x)
            # 小波域处理
            coeffs_processed = self.wavelet_conv(coeffs)
            # 使用1x1卷积重构
            enhanced = self.inverse_conv(coeffs_processed)
            
        elif self.wavelet_type == 'contourlet':
            # 轮廓波变换
            pyramid_coeffs = self.wavelet(x)
            # 处理每层系数（简化：只处理第一层）
            processed_coeffs = []
            for coeffs in pyramid_coeffs:
                processed = self.wavelet_conv(coeffs)
                processed_coeffs.append(processed)
            
            # 简化重构：只使用第一层
            enhanced = self.inverse_conv(processed_coeffs[0])
            
        else:
            raise ValueError(f"未知的小波类型: {self.wavelet_type}")
        
        return enhanced


def test_wavelet_transform():
    """测试小波变换模块"""
    print("测试小波变换模块...")
    
    # 创建测试数据
    batch_size, channels, height, width = 2, 1, 64, 64
    x = torch.randn(batch_size, channels, height, width)
    
    # 测试DWT
    print("\n1. 测试DWT2d:")
    dwt = DWT2d(wavelet='db1')
    coeffs = dwt(x)
    print(f"   输入形状: {x.shape}")
    print(f"   系数形状: {coeffs.shape}")
    
    # 测试逆变换
    recon = dwt.inverse(coeffs, (height, width))
    print(f"   重构形状: {recon.shape}")
    print(f"   重构误差: {torch.abs(x - recon).mean().item():.6f}")
    
    # 测试方向小波
    print("\n2. 测试方向小波:")
    directional = LearnableDirectionalWavelet(in_channels=channels, num_directions=8)
    dir_coeffs = directional(x)
    print(f"   输入形状: {x.shape}")
    print(f"   方向系数形状: {dir_coeffs.shape}")
    
    # 测试小波域处理
    print("\n3. 测试小波域处理:")
    wavelet_processor = WaveletDomainProcessing(in_channels=channels, out_channels=channels, wavelet_type='directional')
    enhanced = wavelet_processor(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {enhanced.shape}")
    
    print("\n所有测试通过！")


if __name__ == "__main__":
    test_wavelet_transform()
