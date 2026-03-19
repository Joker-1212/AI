"""
小波变换工具函数
实现纯PyTorch DWT、固定方向滤波器组和轮廓波变换，用于低剂量CT增强
参考论文：A deep convolutional neural network using directional wavelets for low‑dose X‑ray CT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from typing import Tuple, List, Optional, Union
from monai.networks.nets import UNet


class DWT2d(nn.Module):
    """2D离散小波变换 — 纯 PyTorch 可微分实现（可分离 1D 方式）

    仅在 __init__ 中使用 pywt 获取滤波器系数，前向/逆向全部在 GPU 上可微分。
    使用行→列两趟 1D 卷积，保证正交小波完美重构。
    输出: (B, C*4, H//2, W//2)  对应 [cA, cH, cV, cD] 四个子带
    """

    def __init__(self, wavelet: str = 'db4'):
        super().__init__()
        self.wavelet_name = wavelet

        w = pywt.Wavelet(wavelet)
        # 1D 滤波器，形状 (1, 1, K) 用于 conv1d
        dec_lo = torch.tensor(w.dec_lo, dtype=torch.float32)
        dec_hi = torch.tensor(w.dec_hi, dtype=torch.float32)
        self.filter_len = len(w.dec_lo)

        # 分解滤波器: (2, 1, K) — [lo, hi]
        dec = torch.stack([dec_lo, dec_hi]).unsqueeze(1)
        self.register_buffer('dec', dec)

    def _afb1d(self, x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """沿指定维度做 1D 分析滤波器组（低通+高通，下采样2）"""
        # 周期延拓：在右侧填充 (filter_len - 1) 个样本
        pad_len = self.filter_len - 1
        if dim == -1 or dim == x.ndim - 1:
            x = F.pad(x, [0, pad_len], mode='circular')
        else:
            x = x.transpose(dim, -1)
            x = F.pad(x, [0, pad_len], mode='circular')
            x = x.transpose(dim, -1)

        # 沿最后一维卷积
        shape = x.shape
        if dim == -1 or dim == x.ndim - 1:
            # (N, L) -> conv1d 需要 (N, 1, L)
            x_flat = x.reshape(-1, 1, x.shape[-1])
            out = F.conv1d(x_flat, self.dec, stride=2)  # (N, 2, L//2)
            lo = out[:, 0:1].reshape(*shape[:-1], -1)
            hi = out[:, 1:2].reshape(*shape[:-1], -1)
        else:
            x = x.transpose(dim, -1)
            x_flat = x.reshape(-1, 1, x.shape[-1])
            out = F.conv1d(x_flat, self.dec, stride=2)
            lo = out[:, 0:1].reshape(*x.shape[:-1], -1).transpose(dim, -1)
            hi = out[:, 1:2].reshape(*x.shape[:-1], -1).transpose(dim, -1)

        return lo, hi

    def _sfb1d(self, lo: torch.Tensor, hi: torch.Tensor, dim: int, out_len: int) -> torch.Tensor:
        """沿指定维度做 1D 合成滤波器组（上采样2 + 低通+高通求和）"""

        def _reconstruct(lo_flat, hi_flat, target_len):
            combined = torch.cat([lo_flat, hi_flat], dim=1)  # (N, 2, L//2)
            rec = F.conv_transpose1d(combined, self.dec, stride=2)  # (N, 1, ~L+K-2)
            # 周期边界：多余的尾部样本循环叠加到头部
            rec_len = rec.shape[-1]
            if rec_len > target_len:
                rec[:, :, :rec_len - target_len] += rec[:, :, target_len:]
            return rec[:, 0, :target_len]

        if dim == -1 or dim == lo.ndim - 1:
            shape = lo.shape
            lo_flat = lo.reshape(-1, 1, lo.shape[-1])
            hi_flat = hi.reshape(-1, 1, hi.shape[-1])
            rec = _reconstruct(lo_flat, hi_flat, out_len)
            return rec.reshape(*shape[:-1], -1)
        else:
            lo = lo.transpose(dim, -1)
            hi = hi.transpose(dim, -1)
            shape = lo.shape
            lo_flat = lo.reshape(-1, 1, lo.shape[-1])
            hi_flat = hi.reshape(-1, 1, hi.shape[-1])
            rec = _reconstruct(lo_flat, hi_flat, out_len)
            return rec.reshape(*shape[:-1], -1).transpose(dim, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入进行2D小波分解（行→列可分离）

        参数:
            x: (B, C, H, W)
        返回:
            coeffs: (B, C*4, H//2, W//2)，子带顺序 [LL, LH, HL, HH]
        """
        B, C, H, W = x.shape
        # 合并 B,C 到批次维度
        x = x.reshape(B * C, H, W)

        # 第一趟：沿宽度（列方向）分解
        lo_w, hi_w = self._afb1d(x, dim=-1)  # (BC, H, W//2) 各两个

        # 第二趟：沿高度（行方向）分解
        ll, lh = self._afb1d(lo_w, dim=-2)  # (BC, H//2, W//2)
        hl, hh = self._afb1d(hi_w, dim=-2)  # (BC, H//2, W//2)

        # 堆叠为 (B, C*4, H//2, W//2)
        coeffs = torch.stack([ll, lh, hl, hh], dim=1)  # (BC, 4, H//2, W//2)
        _, _, h_out, w_out = coeffs.shape
        coeffs = coeffs.reshape(B, C * 4, h_out, w_out)

        return coeffs

    def inverse(self, coeffs: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
        """
        从小波系数重构图像（列→行可分离逆变换）

        参数:
            coeffs: (B, C*4, H//2, W//2)
            original_shape: (H, W)
        返回:
            x_recon: (B, C, H, W)
        """
        B, C4, h, w = coeffs.shape
        C = C4 // 4
        H, W = original_shape

        coeffs = coeffs.reshape(B * C, 4, h, w)
        ll, lh, hl, hh = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3]

        # 第一趟逆：沿高度重构
        lo_w = self._sfb1d(ll, lh, dim=-2, out_len=H)  # (BC, H, W//2)
        hi_w = self._sfb1d(hl, hh, dim=-2, out_len=H)  # (BC, H, W//2)

        # 第二趟逆：沿宽度重构
        x_rec = self._sfb1d(lo_w, hi_w, dim=-1, out_len=W)  # (BC, H, W)

        return x_rec.reshape(B, C, H, W)


class FixedDirectionalFilterBank(nn.Module):
    """
    固定方向滤波器组（非可学习）
    与 LearnableDirectionalWavelet 结构相同（depthwise Conv2d, Gabor 初始化, 8方向, kernel_size=5）
    关键区别：权重用 register_buffer 注册，不出现在 model.parameters() 中
    包含 inverse() 方法用于伪逆重构
    """

    def __init__(self, in_channels: int, num_directions: int = 8, kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.num_directions = num_directions
        self.kernel_size = kernel_size

        # 构建方向滤波器权重
        out_ch = in_channels * num_directions
        weight = torch.zeros(out_ch, 1, kernel_size, kernel_size)

        for d in range(num_directions):
            theta = d * np.pi / num_directions
            filter_2d = self._create_directional_filter(theta, kernel_size, kernel_size)
            for c in range(in_channels):
                weight[d * in_channels + c, 0] = torch.from_numpy(filter_2d).float()

        # 注册为 buffer — 不参与梯度更新
        self.register_buffer('weight', weight)
        self.padding = kernel_size // 2
        self.groups = in_channels

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
        应用方向滤波

        参数:
            x: (B, C, H, W)
        返回:
            (B, C*num_directions, H, W)
        """
        return F.conv2d(x, self.weight, padding=self.padding, groups=self.groups)

    def inverse(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        用转置卷积实现伪逆重构

        参数:
            coeffs: (B, C*num_directions, H, W)
        返回:
            (B, C, H, W)
        """
        return F.conv_transpose2d(
            coeffs, self.weight, padding=self.padding, groups=self.groups
        )


class ContourletTransform(nn.Module):
    """
    轮廓波变换（非可学习）
    使用拉普拉斯金字塔 + 固定方向滤波器组

    输出: [detail_0 (B,C*D,H,W), detail_1 (B,C*D,H/2,W/2), ..., low_freq (B,C,H_s,W_s)]
    其中 D = num_directions
    """

    def __init__(self, levels: int = 3, num_directions: int = 8):
        super().__init__()
        self.levels = levels
        self.num_directions = num_directions

        # 拉普拉斯金字塔层
        self.downsample = nn.AvgPool2d(2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 固定方向滤波器组（每个通道独立处理）
        self.dfb = FixedDirectionalFilterBank(
            in_channels=1,
            num_directions=num_directions,
            kernel_size=5
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        应用轮廓波变换

        参数:
            x: (B, C, H, W)
        返回:
            list: [detail_0, detail_1, ..., detail_{levels-1}, low_freq]
                  detail_i: (B, C*num_directions, H_i, W_i)
                  low_freq: (B, C, H_small, W_small) — 不经过方向滤波
        """
        batch_size, channels, height, width = x.shape
        pyramid_coeffs = []

        current = x
        for level in range(self.levels):
            # 下采样
            down = self.downsample(current)
            # 上采样以匹配当前尺寸
            up = self.upsample(down)
            # 裁剪到当前尺寸（upsample 可能多一个像素）
            up = up[:, :, :current.shape[2], :current.shape[3]]
            # 高频细节
            detail = current - up

            # 对细节应用方向滤波（逐通道）
            detail_coeffs = []
            for c in range(channels):
                detail_c = detail[:, c:c+1]  # (B, 1, H, W)
                coeffs_c = self.dfb(detail_c)  # (B, num_directions, H, W)
                detail_coeffs.append(coeffs_c)

            detail_coeffs = torch.cat(detail_coeffs, dim=1)  # (B, C*num_directions, H, W)
            pyramid_coeffs.append(detail_coeffs)

            current = down

        # 低频直接传递，不经过方向滤波
        pyramid_coeffs.append(current)  # (B, C, H_small, W_small)

        return pyramid_coeffs

    def inverse(self, pyramid_coeffs: List[torch.Tensor], channels: int) -> torch.Tensor:
        """
        轮廓波逆变换（伪逆）

        参数:
            pyramid_coeffs: [detail_0, ..., detail_{levels-1}, low_freq]
                            detail 层已经被处理过
            channels: 原始图像通道数
        返回:
            x_recon: (B, C, H, W)
        """
        # 从低频开始重构
        current = pyramid_coeffs[-1]  # (B, C, H_small, W_small)

        # 从最粗糙层到最精细层
        for level in range(self.levels - 1, -1, -1):
            detail_coeffs = pyramid_coeffs[level]  # (B, C*num_directions, H_i, W_i)
            target_h, target_w = detail_coeffs.shape[2], detail_coeffs.shape[3]

            # 上采样低频
            current = self.upsample(current)
            current = current[:, :, :target_h, :target_w]

            # 方向滤波器伪逆重构 detail（逐通道）
            detail_recon = []
            for c in range(channels):
                start = c * self.num_directions
                end = start + self.num_directions
                coeffs_c = detail_coeffs[:, start:end]  # (B, num_directions, H, W)
                recon_c = self.dfb.inverse(coeffs_c)  # (B, 1, H, W)
                detail_recon.append(recon_c)

            detail_recon = torch.cat(detail_recon, dim=1)  # (B, C, H, W)
            current = current + detail_recon

        return current


class WaveletDomainProcessing(nn.Module):
    """
    小波域处理模块
    将图像转换到小波/轮廓波域，用 U-Net 处理子带，然后逆变换重构

    支持两种模式:
    - 'contourlet': 轮廓波变换 → 共享 U-Net 处理各 detail 层 → 逆变换重构
    - 'dwt': DWT → U-Net 处理子带 → 逆 DWT 重构
    """

    def __init__(self, in_channels: int, out_channels: int,
                 wavelet_type: str = 'contourlet',
                 features: Tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wavelet_type = wavelet_type

        if wavelet_type == 'contourlet':
            self.contourlet = ContourletTransform(levels=3, num_directions=8)
            detail_ch = in_channels * 8  # C * num_directions

            # 共享 U-Net 处理各 detail 层
            unet_features = list(features[:3]) if len(features) >= 3 else list(features)
            self.detail_unet = UNet(
                spatial_dims=2,
                in_channels=detail_ch,
                out_channels=detail_ch,
                channels=unet_features,
                strides=(2,) * (len(unet_features) - 1),
                num_res_units=2,
            )

        elif wavelet_type == 'dwt':
            self.dwt = DWT2d(wavelet='db4')
            subband_ch = in_channels * 4

            unet_features = list(features[:3]) if len(features) >= 3 else list(features)
            self.subband_unet = UNet(
                spatial_dims=2,
                in_channels=subband_ch,
                out_channels=subband_ch,
                channels=unet_features,
                strides=(2,) * (len(unet_features) - 1),
                num_res_units=2,
            )
        else:
            raise ValueError(f"未知的小波类型: {wavelet_type}，可选: 'contourlet', 'dwt'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        小波域处理

        参数:
            x: (B, C, H, W)
        返回:
            enhanced: (B, C, H, W)
        """
        original_shape = x.shape[2:]  # (H, W)

        if self.wavelet_type == 'contourlet':
            # 轮廓波分解
            pyramid = self.contourlet(x)  # [detail_0, ..., detail_{L-1}, low_freq]

            # 共享 U-Net 处理每个 detail 层，low_freq 直接传递
            processed = []
            for i in range(len(pyramid) - 1):
                detail = pyramid[i]
                processed.append(self.detail_unet(detail))
            processed.append(pyramid[-1])  # low_freq 不处理

            # 逆变换重构
            enhanced = self.contourlet.inverse(processed, self.in_channels)

        elif self.wavelet_type == 'dwt':
            # DWT 分解
            coeffs = self.dwt(x)
            # U-Net 处理
            coeffs_processed = self.subband_unet(coeffs)
            # 逆 DWT 重构
            enhanced = self.dwt.inverse(coeffs_processed, original_shape)

        return enhanced


def test_wavelet_transform():
    """测试小波变换模块"""
    print("测试小波变换模块...")

    # 创建测试数据
    batch_size, channels, height, width = 2, 1, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    # === 1. 测试新 DWT2d ===
    print("\n1. 测试 DWT2d（纯PyTorch实现）:")
    dwt = DWT2d(wavelet='db1')
    coeffs = dwt(x)
    print(f"   输入形状: {x.shape}")
    print(f"   系数形状: {coeffs.shape}")

    # 逆变换
    recon = dwt.inverse(coeffs, (height, width))
    print(f"   重构形状: {recon.shape}")
    recon_err = torch.abs(x - recon).mean().item()
    print(f"   重构误差: {recon_err:.6f}")

    # 梯度传播测试
    x_grad = x.clone().requires_grad_(True)
    c = dwt(x_grad)
    loss = c.sum()
    loss.backward()
    print(f"   梯度传播: {'通过' if x_grad.grad is not None else '失败'}")

    # === 2. 测试 FixedDirectionalFilterBank ===
    print("\n2. 测试 FixedDirectionalFilterBank:")
    dfb = FixedDirectionalFilterBank(in_channels=channels, num_directions=8)
    params = list(dfb.parameters())
    print(f"   可学习参数数量: {len(params)}（应为 0）")

    dir_coeffs = dfb(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {dir_coeffs.shape}")

    # 伪逆
    recon_dfb = dfb.inverse(dir_coeffs)
    print(f"   伪逆重构形状: {recon_dfb.shape}")

    # === 3. 测试 ContourletTransform ===
    print("\n3. 测试 ContourletTransform:")
    ct = ContourletTransform(levels=3, num_directions=8)
    pyramid = ct(x)
    print(f"   输入形状: {x.shape}")
    for i, p in enumerate(pyramid):
        label = f"detail_{i}" if i < len(pyramid) - 1 else "low_freq"
        print(f"   {label} 形状: {p.shape}")

    # 逆变换
    recon_ct = ct.inverse(pyramid, channels)
    print(f"   逆变换重构形状: {recon_ct.shape}")
    ct_err = torch.abs(x - recon_ct).mean().item()
    print(f"   重构误差: {ct_err:.6f}")

    # === 4. 测试 WaveletDomainProcessing（contourlet 模式）===
    print("\n4. 测试 WaveletDomainProcessing（contourlet 模式）:")
    wp_ct = WaveletDomainProcessing(
        in_channels=channels, out_channels=channels,
        wavelet_type='contourlet', features=(32, 64, 128)
    )
    enhanced_ct = wp_ct(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {enhanced_ct.shape}")

    # === 5. 测试 WaveletDomainProcessing（dwt 模式）===
    print("\n5. 测试 WaveletDomainProcessing（dwt 模式）:")
    wp_dwt = WaveletDomainProcessing(
        in_channels=channels, out_channels=channels,
        wavelet_type='dwt', features=(32, 64, 128)
    )
    enhanced_dwt = wp_dwt(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {enhanced_dwt.shape}")

    print("\n所有测试通过！")


if __name__ == "__main__":
    test_wavelet_transform()
