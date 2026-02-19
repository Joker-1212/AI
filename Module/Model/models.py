"""
低剂量CT增强模型
"""
import torch
import torch.nn as nn
import monai
from monai.networks.nets import UNet, AttentionUnet, ResNet, DenseNet
from monai.networks.blocks import Convolution, ResidualUnit
from typing import Tuple, Optional, Union

from ..Config.config import ModelConfig
from ..Tools.wavelet_transform import WaveletDomainProcessing, LearnableDirectionalWavelet, DWT2d


class CTEnhancementModel(nn.Module):
    """低剂量CT增强模型基类"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
    
    def forward(self, x):
        raise NotImplementedError


class UNet3DModel(CTEnhancementModel):
    """3D UNet模型 - 针对深度维度为1的优化版本（实际使用2D UNet）"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # 对于深度维度为1的数据，使用2D UNet但保持3D张量形状
        features = config.features
        if len(features) > 4:
            features = features[:4]
        
        num_strides = len(features) - 1
        strides = (2,) * num_strides
        
        self.model = UNet(
            spatial_dims=2,  # 使用2D卷积
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=features,
            strides=strides,
            num_res_units=2,
            dropout=config.dropout,
            norm="batch" if config.use_batch_norm else None,
            act=config.activation.lower(),
        )
    
    def forward(self, x):
        # 输入形状: [B, C, H, W, D=1]
        # 转换为2D: [B, C, H, W]
        x_2d = x.squeeze(-1)
        output_2d = self.model(x_2d)
        # 恢复深度维度
        return output_2d.unsqueeze(-1)


class UNet2DModel(CTEnhancementModel):
    """2D UNet模型 - 用于2D数据"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # 对于channels=(32, 64, 128, 256)，需要3个strides
        # 计算strides：每个下采样步骤
        num_strides = len(config.features) - 1
        strides = (2,) * num_strides  # 例如：(2, 2, 2)
        
        self.model = UNet(
            spatial_dims=2,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=config.features,
            strides=strides,
            num_res_units=2,
            dropout=config.dropout,
            norm="batch" if config.use_batch_norm else None,
            act=config.activation.lower(),
        )
    
    def forward(self, x):
        # 输入形状: [batch, channel, height, width, depth]
        # 对于2D模型，我们需要将深度维度合并到批次或通道中
        # 最简单的方法：取深度维度的平均值
        if x.shape[-1] > 1:
            # 如果深度维度>1，取平均值
            x = x.mean(dim=-1, keepdim=True)
        # 移除深度维度: [B, C, H, W, 1] -> [B, C, H, W]
        x = x.squeeze(-1)
        return self.model(x).unsqueeze(-1)  # 添加回深度维度


class AttentionUNetModel(CTEnhancementModel):
    """注意力UNet模型 - 针对深度维度为1的优化版本（实际使用2D AttentionUnet）"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # 使用2D AttentionUnet
        features = config.features
        if len(features) > 4:
            features = features[:4]
        
        num_strides = len(features) - 1
        strides = (2,) * num_strides
        
        self.model = AttentionUnet(
            spatial_dims=2,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=features,
            strides=strides,
            kernel_size=3,
            up_kernel_size=3,
            dropout=config.dropout,
        )
    
    def forward(self, x):
        # 输入形状: [B, C, H, W, D=1]
        # 转换为2D: [B, C, H, W]
        x_2d = x.squeeze(-1)
        output_2d = self.model(x_2d)
        # 恢复深度维度
        return output_2d.unsqueeze(-1)


class ResUNetModel(CTEnhancementModel):
    """残差UNet模型 - 针对深度维度为1的优化版本（实际使用2D UNet）"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # 对于深度维度为1的数据，使用2D UNet但保持3D张量形状
        features = config.features
        if len(features) > 4:
            features = features[:4]
        
        num_strides = len(features) - 1
        strides = (2,) * num_strides
        
        # 使用MONAI的UNet并启用残差单元
        self.model = UNet(
            spatial_dims=2,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=features,
            strides=strides,
            num_res_units=4,  # 更多残差单元
            dropout=config.dropout,
            norm="batch" if config.use_batch_norm else None,
            act=config.activation.lower(),
        )
    
    def forward(self, x):
        # 输入形状: [B, C, H, W, D=1]
        # 转换为2D: [B, C, H, W]
        x_2d = x.squeeze(-1)
        output_2d = self.model(x_2d)
        # 恢复深度维度
        return output_2d.unsqueeze(-1)


class DenseUNetModel(CTEnhancementModel):
    """密集连接UNet模型 - 使用MONAI的DenseNet（2D版本）"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # 使用MONAI的DenseNet作为基础
        from monai.networks.nets import DenseNet
        
        # 计算合适的growth_rate和block_config
        # 简化配置：使用较小的网络
        init_features = config.features[0]
        growth_rate = 16
        block_config = (4, 4, 4, 4)  # 4个密集块，每个4层
        
        # 使用num_classes=None来获取特征图而不是分类得分
        self.densenet = DenseNet(
            spatial_dims=2,
            in_channels=config.in_channels,
            out_channels=init_features,  # 输出特征通道数
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            bn_size=4,
            act=config.activation.lower(),
            norm="batch" if config.use_batch_norm else None,
            dropout_prob=config.dropout,
            num_classes=None,  # 关键：返回特征图
        )
        
        # 将特征图转换为输出通道
        self.final_conv = nn.Conv2d(init_features, config.out_channels, kernel_size=1)
    
    def forward(self, x):
        # 输入形状: [B, C, H, W, D=1]
        # 转换为2D: [B, C, H, W]
        x_2d = x.squeeze(-1)
        features = self.densenet(x_2d)
        output_2d = self.final_conv(features)
        # 恢复深度维度
        return output_2d.unsqueeze(-1)


class MultiScaleModel(CTEnhancementModel):
    """改进的多尺度模型，使用U-Net进行多分辨率分解，添加残差连接和跳跃连接"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # 使用2D U-Net，因为深度维度为1
        features = config.features
        if len(features) > 4:
            features = features[:4]
        
        # 三个尺度：原始、1/2、1/4
        self.downsample2 = nn.AvgPool2d(2, stride=2)
        self.downsample4 = nn.AvgPool2d(4, stride=4)
        
        # 原始尺度UNet
        self.unet_original = UNet(
            spatial_dims=2,
            in_channels=config.in_channels,
            out_channels=features[0],
            channels=features,
            strides=(2, 2, 2),
            num_res_units=2,
            dropout=config.dropout,
            norm="batch" if config.use_batch_norm else None,
            act=config.activation.lower(),
        )
        
        # 1/2尺度UNet
        self.unet_half = UNet(
            spatial_dims=2,
            in_channels=config.in_channels,
            out_channels=features[0],
            channels=[f//2 for f in features],
            strides=(2, 2, 2),
            num_res_units=2,
            dropout=config.dropout,
            norm="batch" if config.use_batch_norm else None,
            act=config.activation.lower(),
        )
        
        # 1/4尺度UNet
        self.unet_quarter = UNet(
            spatial_dims=2,
            in_channels=config.in_channels,
            out_channels=features[0],
            channels=[f//4 for f in features],
            strides=(2, 2, 2),
            num_res_units=1,
            dropout=config.dropout,
            norm="batch" if config.use_batch_norm else None,
            act=config.activation.lower(),
        )
        
        # 特征融合模块，包含残差连接
        self.fusion = nn.Sequential(
            nn.Conv2d(features[0] * 3, features[0] * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0] * 2, features[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
        )
        
        # 输出层，添加残差连接
        self.output = nn.Sequential(
            nn.Conv2d(features[0] + config.in_channels, features[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], config.out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        # 输入形状: [B, C, H, W, D=1]
        x_2d = x.squeeze(-1)
        
        # 多尺度输入
        x_half = self.downsample2(x_2d)
        x_quarter = self.downsample4(x_2d)
        
        # 各尺度特征提取
        feat_original = self.unet_original(x_2d)
        feat_half = self.unet_half(x_half)
        feat_quarter = self.unet_quarter(x_quarter)
        
        # 上采样特征以匹配原始尺寸
        feat_half_up = torch.nn.functional.interpolate(feat_half, size=feat_original.shape[2:], mode='bilinear', align_corners=True)
        feat_quarter_up = torch.nn.functional.interpolate(feat_quarter, size=feat_original.shape[2:], mode='bilinear', align_corners=True)
        
        # 融合多尺度特征
        fused = torch.cat([feat_original, feat_half_up, feat_quarter_up], dim=1)
        fused = self.fusion(fused)
        
        # 残差连接：将原始输入与融合特征结合
        combined = torch.cat([fused, x_2d], dim=1)
        output_2d = self.output(combined)
        
        # 恢复深度维度
        return output_2d.unsqueeze(-1)


class WaveletDomainCNNModel(CTEnhancementModel):
    """小波域CNN模型 - 在小波域中进行特征提取和处理"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # 小波域处理模块
        self.wavelet_processor = WaveletDomainProcessing(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            wavelet_type='directional'  # 可选: 'directional', 'dwt', 'contourlet'
        )
        
        # 后续的卷积细化
        self.refine = nn.Sequential(
            nn.Conv2d(config.out_channels, config.features[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(config.features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.features[0], config.features[0] // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.features[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.features[0] // 2, config.out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        # 输入形状: [B, C, H, W, D=1]
        # 转换为2D: [B, C, H, W]
        x_2d = x.squeeze(-1)
        
        # 小波域处理
        wavelet_enhanced = self.wavelet_processor(x_2d)
        
        # 细化
        refined = self.refine(wavelet_enhanced)
        
        # 恢复深度维度
        return refined.unsqueeze(-1)


class FBPConvNetModel(CTEnhancementModel):
    """FBPConvNet模型 - 先FBP后CNN去噪"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # 模拟FBP的简单卷积层（实际应用中应使用真实的FBP）
        self.fbp_simulator = nn.Sequential(
            nn.Conv2d(config.in_channels, config.features[0], kernel_size=7, padding=3),
            nn.BatchNorm2d(config.features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.features[0], config.in_channels, kernel_size=7, padding=3),
        )
        
        # 去噪CNN（U-Net风格）
        self.denoiser = UNet(
            spatial_dims=2,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=config.features,
            strides=(2, 2, 2),
            num_res_units=2,
            dropout=config.dropout,
            norm="batch" if config.use_batch_norm else None,
            act=config.activation.lower(),
        )
    
    def forward(self, x):
        # 输入形状: [B, C, H, W, D=1]
        x_2d = x.squeeze(-1)
        
        # 模拟FBP重建
        fbp_recon = self.fbp_simulator(x_2d)
        
        # CNN去噪
        denoised = self.denoiser(fbp_recon)
        
        # 恢复深度维度
        return denoised.unsqueeze(-1)


def create_model(config: ModelConfig) -> CTEnhancementModel:
    """根据配置创建模型"""
    model_map = {
        "UNet3D": UNet3DModel,
        "UNet2D": UNet2DModel,
        "AttentionUNet": AttentionUNetModel,
        "ResUNet": ResUNetModel,
        "DenseUNet": DenseUNetModel,
        "MultiScale": MultiScaleModel,
        "WaveletDomainCNN": WaveletDomainCNNModel,
        "FBPConvNet": FBPConvNetModel,
    }
    
    if config.model_name not in model_map:
        raise ValueError(f"未知模型: {config.model_name}。可选: {list(model_map.keys())}")
    
    return model_map[config.model_name](config)


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    config = ModelConfig()
    model = create_model(config)
    
    print(f"模型: {config.model_name}")
    print(f"参数量: {count_parameters(model):,}")
    
    # 测试前向传播
    dummy_input = torch.randn(1, config.in_channels, 256, 256, 1)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
