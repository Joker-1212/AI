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


class CTEnhancementModel(nn.Module):
    """低剂量CT增强模型基类"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
    
    def forward(self, x):
        raise NotImplementedError


class UNet3DModel(CTEnhancementModel):
    """3D UNet模型"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.model = UNet(
            spatial_dims=3,
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
        return self.model(x)


class AttentionUNetModel(CTEnhancementModel):
    """注意力UNet模型"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.model = AttentionUnet(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=config.features,
            strides=(2, 2, 2),
            dropout=config.dropout,
            norm="batch" if config.use_batch_norm else None,
            act=config.activation.lower(),
        )
    
    def forward(self, x):
        return self.model(x)


class ResUNetModel(CTEnhancementModel):
    """残差UNet模型"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # 使用MONAI的UNet并启用残差单元
        self.model = UNet(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=config.features,
            strides=(2, 2, 2),
            num_res_units=4,  # 更多残差单元
            dropout=config.dropout,
            norm="batch" if config.use_batch_norm else None,
            act=config.activation.lower(),
        )
    
    def forward(self, x):
        return self.model(x)


class DenseUNetModel(CTEnhancementModel):
    """密集连接UNet模型"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # 自定义密集块UNet
        from monai.networks.blocks import DenseBlock
        
        self.encoder1 = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=config.in_channels,
                out_channels=config.features[0],
                kernel_size=3,
                strides=1,
                padding=1,
                norm="batch" if config.use_batch_norm else None,
                act=config.activation.lower(),
            ),
            DenseBlock(
                spatial_dims=3,
                in_channels=config.features[0],
                out_channels=config.features[0],
                kernel_size=3,
                layers=3,
                dropout=config.dropout,
                norm="batch" if config.use_batch_norm else None,
                act=config.activation.lower(),
            )
        )
        
        self.encoder2 = nn.Sequential(
            nn.MaxPool3d(2),
            Convolution(
                spatial_dims=3,
                in_channels=config.features[0],
                out_channels=config.features[1],
                kernel_size=3,
                strides=1,
                padding=1,
                norm="batch" if config.use_batch_norm else None,
                act=config.activation.lower(),
            ),
            DenseBlock(
                spatial_dims=3,
                in_channels=config.features[1],
                out_channels=config.features[1],
                kernel_size=3,
                layers=3,
                dropout=config.dropout,
                norm="batch" if config.use_batch_norm else None,
                act=config.activation.lower(),
            )
        )
        
        # 解码器部分
        self.decoder2 = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=config.features[1] + config.features[0],
                out_channels=config.features[0],
                kernel_size=3,
                strides=1,
                padding=1,
                norm="batch" if config.use_batch_norm else None,
                act=config.activation.lower(),
            ),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )
        
        self.final = Convolution(
            spatial_dims=3,
            in_channels=config.features[0],
            out_channels=config.out_channels,
            kernel_size=1,
            strides=1,
            padding=0,
            norm=None,
            act=None,
        )
    
    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        
        # 解码器
        dec2 = self.decoder2(torch.cat([enc2, enc1], dim=1))
        
        # 最终输出
        out = self.final(dec2)
        return out


class MultiScaleModel(CTEnhancementModel):
    """多尺度模型，结合不同分辨率特征"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # 多尺度输入
        self.downsample2 = nn.AvgPool3d(2)
        self.downsample4 = nn.AvgPool3d(4)
        
        # 主UNet
        self.main_unet = UNet(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.features[0],
            channels=config.features,
            strides=(2, 2, 2),
            num_res_units=2,
            dropout=config.dropout,
            norm="batch" if config.use_batch_norm else None,
            act=config.activation.lower(),
        )
        
        # 辅助UNet（处理下采样特征）
        self.aux_unet = UNet(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.features[0],
            channels=[f//2 for f in config.features],
            strides=(2, 2, 2),
            num_res_units=1,
            dropout=config.dropout,
            norm="batch" if config.use_batch_norm else None,
            act=config.activation.lower(),
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=config.features[0] * 2,
                out_channels=config.features[0],
                kernel_size=3,
                strides=1,
                padding=1,
                norm="batch" if config.use_batch_norm else None,
                act=config.activation.lower(),
            ),
            Convolution(
                spatial_dims=3,
                in_channels=config.features[0],
                out_channels=config.out_channels,
                kernel_size=1,
                strides=1,
                padding=0,
                norm=None,
                act=None,
            )
        )
    
    def forward(self, x):
        # 原始尺度
        main_feat = self.main_unet(x)
        
        # 下采样尺度
        x_down = self.downsample2(x)
        aux_feat = self.aux_unet(x_down)
        
        # 上采样辅助特征以匹配尺寸
        aux_feat = nn.functional.interpolate(
            aux_feat, size=main_feat.shape[2:], mode='trilinear', align_corners=True
        )
        
        # 融合特征
        fused = torch.cat([main_feat, aux_feat], dim=1)
        out = self.fusion(fused)
        
        return out


def create_model(config: ModelConfig) -> CTEnhancementModel:
    """根据配置创建模型"""
    model_map = {
        "UNet3D": UNet3DModel,
        "AttentionUNet": AttentionUNetModel,
        "ResUNet": ResUNetModel,
        "DenseUNet": DenseUNetModel,
        "MultiScale": MultiScaleModel,
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
