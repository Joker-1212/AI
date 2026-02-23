"""
损失函数模块
实现混合损失函数（L1 + SSIM + 感知损失）和多尺度损失计算
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import warnings
import contextlib

# 抑制torchvision相关的弃用警告
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torchvision')
warnings.filterwarnings('ignore', category=FutureWarning, module='torchvision')


class SSIMLoss(nn.Module):
    """SSIM损失函数（负SSIM）"""
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.register_buffer('window', self._create_gaussian_window())
    
    def _create_gaussian_window(self) -> torch.Tensor:
        """创建高斯窗口"""
        from math import exp
        window_1d = torch.tensor([
            exp(-(x - self.window_size // 2) ** 2 / (2 * self.sigma ** 2))
            for x in range(self.window_size)
        ])
        window_1d = window_1d / window_1d.sum()
        window_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)
        return window_2d.unsqueeze(0).unsqueeze(0)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算SSIM损失
        
        参数:
            pred: 预测图像 (B, C, H, W) 或 (B, C, H, W, D)
            target: 目标图像 (B, C, H, W) 或 (B, C, H, W, D)
        
        返回:
            loss: SSIM损失值
        """
        # 处理5D输入（深度维度为1的情况）
        if pred.ndim == 5:
            # 压缩深度维度
            pred = pred.squeeze(-1)
            target = target.squeeze(-1)
        
        batch_size, channels = pred.shape[0], pred.shape[1]
        
        if pred.shape[1] > 1:
            # 多通道：分别计算每个通道
            ssim_values = []
            for c in range(channels):
                pred_c = pred[:, c:c+1]
                target_c = target[:, c:c+1]
                ssim_c = self._ssim_single_channel(pred_c, target_c)
                ssim_values.append(ssim_c)
            ssim = torch.stack(ssim_values).mean()
        else:
            ssim = self._ssim_single_channel(pred, target)
        
        # SSIM损失 = 1 - SSIM
        return 1.0 - ssim
    
    def _ssim_single_channel(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """单通道SSIM计算"""
        window = self.window.to(pred.device)
        
        # 计算均值
        mu1 = F.conv2d(pred, window, padding=self.window_size//2, groups=1)
        mu2 = F.conv2d(target, window, padding=self.window_size//2, groups=1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # 计算方差
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size//2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size//2, groups=1) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size//2, groups=1) - mu1_mu2
        
        # SSIM公式
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
        
        return ssim_map.mean()


class PerceptualLoss(nn.Module):
    """
    感知损失 - 使用预训练的VGG网络（简化版）
    
    使用预训练的VGG19网络提取特征，计算特征空间中的L1损失。
    支持torchvision 0.13+的新API（weights参数）和旧版API（pretrained参数）的向后兼容性。
    
    注意:
        - torchvision >= 0.13: 使用新的weights参数API
        - torchvision < 0.13: 使用已弃用的pretrained参数（向后兼容）
        - 如果无法加载预训练权重，将使用未初始化的模型并发出警告
    """
    
    def __init__(self, layer_indices: Tuple[int, ...] = (1, 6, 11, 20)):
        """
        初始化感知损失函数
        
        参数:
            layer_indices: VGG19特征层索引（使用ReLU层）
                relu1_2: 1, relu2_2: 6, relu3_2: 11, relu4_2: 20
                
        版本兼容性:
            自动检测torchvision版本并使用相应的API加载预训练权重：
            - torchvision 0.13+: 使用models.vgg19(weights=models.VGG19_Weights.DEFAULT)
            - torchvision < 0.13: 使用models.vgg19(pretrained=True)（已弃用但向后兼容）
        """
        super().__init__()
        # 加载预训练的VGG19（兼容torchvision新旧版本）
        from torchvision import models
        
        # 版本检测和兼容性处理
        vgg = self._load_vgg19_with_compatibility()
        
        # 提取指定层的特征提取器
        self.layers = nn.ModuleList()
        for idx in layer_indices:
            self.layers.append(vgg[:idx+1])  # 包含该层及之前的所有层
        
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False
    
    def _load_vgg19_with_compatibility(self):
        """
        加载VGG19模型，兼容torchvision新旧版本API
        
        Returns:
            VGG19模型的特征提取器
        """
        # 在导入torchvision之前设置警告过滤器
        import warnings
        with warnings.catch_warnings():
            # 抑制所有UserWarning、DeprecationWarning和FutureWarning
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)
            
            from torchvision import models
            
            try:
                # 尝试torchvision 0.13+的新API
                # 首先检查是否有VGG19_Weights枚举
                if hasattr(models, 'VGG19_Weights'):
                    # 使用新的weights参数，优先使用DEFAULT权重
                    try:
                        # 尝试使用DEFAULT权重（torchvision 0.13+推荐）
                        weights = models.VGG19_Weights.DEFAULT
                    except AttributeError:
                        # 回退到IMAGENET1K_V1
                        weights = models.VGG19_Weights.IMAGENET1K_V1
                    
                    vgg = models.vgg19(weights=weights).features
                else:
                    # 回退到旧版pretrained参数
                    vgg = models.vgg19(pretrained=True).features
                    warnings.warn(
                        "使用已弃用的pretrained参数加载VGG19权重。请升级到torchvision 0.13+。",
                        DeprecationWarning,
                        stacklevel=2
                    )
            except (AttributeError, TypeError) as e:
                # 如果上述方法都失败，尝试直接使用pretrained参数
                try:
                    vgg = models.vgg19(pretrained=True).features
                    warnings.warn(
                        f"回退到旧版API加载VGG19权重: {e}",
                        DeprecationWarning,
                        stacklevel=2
                    )
                except Exception as e2:
                    # 如果仍然失败，创建未初始化的模型
                    warnings.warn(
                        f"无法加载预训练权重，使用未初始化的VGG19: {e2}",
                        RuntimeWarning,
                        stacklevel=2
                    )
                    vgg = models.vgg19().features
        
        return vgg
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算感知损失
        
        参数:
            pred: 预测图像 (B, C, H, W) 或 (B, C, H, W, D)，假设C=3或1
            target: 目标图像 (B, C, H, W) 或 (B, C, H, W, D)
        
        返回:
            loss: 感知损失值
        """
        # 处理5D输入（深度维度为1的情况）
        if pred.ndim == 5:
            # 压缩深度维度
            pred = pred.squeeze(-1)
            target = target.squeeze(-1)
        
        # 确保模型在正确的设备上
        self.to(pred.device)
        
        # 如果输入是单通道，复制到3通道
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        # 归一化到ImageNet统计量
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        # 提取特征并计算损失
        loss = 0.0
        for layer in self.layers:
            pred_feat = layer(pred_norm)
            target_feat = layer(target_norm)
            loss += F.l1_loss(pred_feat, target_feat)
        
        return loss / len(self.layers)


class MixedLoss(nn.Module):
    """混合损失函数：L1 + SSIM + 感知损失"""
    
    def __init__(self, weights: Tuple[float, float, float] = (1.0, 0.5, 0.1)):
        super().__init__()
        self.weights = weights
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = PerceptualLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 处理5D输入（深度维度为1的情况）
        if pred.ndim == 5:
            # 压缩深度维度
            pred = pred.squeeze(-1)
            target = target.squeeze(-1)
        
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        
        return (self.weights[0] * l1 +
                self.weights[1] * ssim +
                self.weights[2] * perceptual)


class MultiScaleLoss(nn.Module):
    """多尺度损失计算"""
    
    def __init__(self, base_loss: str = "L1Loss", scales: Tuple[int, ...] = (1, 2, 4),
                 weights: Tuple[float, ...] = (1.0, 0.5, 0.25)):
        super().__init__()
        self.scales = scales
        self.weights = weights
        
        # 基础损失函数
        if base_loss == "L1Loss":
            self.base_loss = nn.L1Loss()
        elif base_loss == "MSELoss":
            self.base_loss = nn.MSELoss()
        elif base_loss == "MixedLoss":
            self.base_loss = MixedLoss()
        else:
            raise ValueError(f"未知的基础损失函数: {base_loss}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算多尺度损失
        
        参数:
            pred: 预测图像 (B, C, H, W) 或 (B, C, H, W, D)
            target: 目标图像 (B, C, H, W) 或 (B, C, H, W, D)
        
        返回:
            loss: 多尺度损失值
        """
        # 处理5D输入（深度维度为1的情况）
        if pred.ndim == 5:
            # 压缩深度维度
            pred = pred.squeeze(-1)
            target = target.squeeze(-1)
        
        total_loss = 0.0
        
        for scale, weight in zip(self.scales, self.weights):
            if scale == 1:
                pred_scaled = pred
                target_scaled = target
            else:
                # 下采样
                size = (pred.shape[2] // scale, pred.shape[3] // scale)
                pred_scaled = F.interpolate(pred, size=size, mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, size=size, mode='bilinear', align_corners=False)
            
            loss = self.base_loss(pred_scaled, target_scaled)
            total_loss += weight * loss
        
        return total_loss / sum(self.weights)


def create_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """创建损失函数"""
    if loss_name == "L1Loss":
        return nn.L1Loss()
    elif loss_name == "MSELoss":
        return nn.MSELoss()
    elif loss_name == "SmoothL1Loss":
        return nn.SmoothL1Loss()
    elif loss_name == "SSIMLoss":
        return SSIMLoss(**kwargs)
    elif loss_name == "MixedLoss":
        weights = kwargs.get('weights', (1.0, 0.5, 0.1))
        return MixedLoss(weights=weights)
    elif loss_name == "MultiScaleLoss":
        base_loss = kwargs.get('base_loss', 'L1Loss')
        scales = kwargs.get('scales', (1, 2, 4))
        weights = kwargs.get('weights', (1.0, 0.5, 0.25))
        return MultiScaleLoss(base_loss=base_loss, scales=scales, weights=weights)
    else:
        raise ValueError(f"未知的损失函数: {loss_name}")


if __name__ == "__main__":
    # 测试损失函数
    print("测试损失函数...")
    
    batch_size, channels, height, width = 2, 1, 64, 64
    pred = torch.randn(batch_size, channels, height, width)
    target = torch.randn(batch_size, channels, height, width)
    
    # 测试SSIM损失
    ssim_loss = SSIMLoss()
    loss_ssim = ssim_loss(pred, target)
    print(f"SSIM损失: {loss_ssim.item():.6f}")
    
    # 测试混合损失
    mixed_loss = MixedLoss(weights=(1.0, 0.5, 0.1))
    loss_mixed = mixed_loss(pred, target)
    print(f"混合损失: {loss_mixed.item():.6f}")
    
    # 测试多尺度损失
    multi_loss = MultiScaleLoss(base_loss="L1Loss")
    loss_multi = multi_loss(pred, target)
    print(f"多尺度损失: {loss_multi.item():.6f}")
    
    print("所有损失函数测试通过！")
