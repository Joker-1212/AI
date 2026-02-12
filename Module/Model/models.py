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
    """3D UNet模型 - 优化处理深度维度为1的情况"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # 对于深度维度为1的数据，使用strides=(2, 2, 1)避免深度维度下采样
        # 同时调整channels数量以确保strides长度匹配
        # strides长度应该等于len(channels) - 1
        features = config.features
        if len(features) > 3:
            # 如果特征层太多，减少层数以避免深度维度问题
            features = features[:3]
        
        # 确保strides长度匹配
        strides = (2, 2, 1)  # 深度维度不下采样
        if len(features) - 1 < len(strides):
            # 如果特征层数不足，减少strides长度
            strides = strides[:len(features) - 1]
        
        self.model = UNet(
            spatial_dims=3,
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
        # 检查输入深度维度
        input_depth = x.shape[-1]
        if input_depth == 1:
            # 深度维度为1，模型应该能处理
            # 但可能需要特殊处理
            pass
        return self.model(x)


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
    """注意力UNet模型"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # AttentionUnet没有norm和act参数
        # 使用默认的kernel_size和up_kernel_size
        self.model = AttentionUnet(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=config.features,
            strides=(2, 2, 1),  # 深度维度不下采样
            kernel_size=3,
            up_kernel_size=3,
            dropout=config.dropout,
        )
    
    def forward(self, x):
        return self.model(x)


class ResUNetModel(CTEnhancementModel):
    """残差UNet模型 - 优化处理深度维度为1的情况"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # 对于深度维度为1的数据，使用strides=(2, 2, 1)避免深度维度下采样
        # 调整特征层数以匹配strides
        features = config.features
        if len(features) > 3:
            features = features[:3]
        
        strides = (2, 2, 1)  # 深度维度不下采样
        if len(features) - 1 < len(strides):
            strides = strides[:len(features) - 1]
        
        # 使用MONAI的UNet并启用残差单元
        self.model = UNet(
            spatial_dims=3,
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
        return self.model(x)


class DenseUNetModel(CTEnhancementModel):
    """密集连接UNet模型 - 使用MONAI的DenseNet"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # 使用MONAI的DenseNet作为基础
        from monai.networks.nets import DenseNet
        
        # 计算合适的growth_rate和block_config
        # 简化配置：使用较小的网络
        init_features = config.features[0]
        growth_rate = 16
        block_config = (4, 4, 4, 4)  # 4个密集块，每个4层
        
        self.model = DenseNet(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            bn_size=4,
            act=config.activation.lower(),
            norm="batch" if config.use_batch_norm else None,
            dropout_prob=config.dropout,
        )
    
    def forward(self, x):
        return self.model(x)


class MultiScaleModel(CTEnhancementModel):
    """多尺度模型，结合不同分辨率特征 - 优化处理深度维度为1的情况"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # 调整特征层数以匹配深度维度
        features = config.features
        if len(features) > 3:
            features = features[:3]
        
        strides = (2, 2, 1)  # 深度维度不下采样
        if len(features) - 1 < len(strides):
            strides = strides[:len(features) - 1]
        
        # 多尺度输入 - 对于深度维度为1，避免深度下采样
        self.downsample2 = nn.AvgPool3d((2, 2, 1))  # 只在空间维度下采样
        self.downsample4 = nn.AvgPool3d((4, 4, 1))  # 只在空间维度下采样
        
        # 主UNet
        self.main_unet = UNet(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.features[0],
            channels=features,
            strides=strides,
            num_res_units=2,
            dropout=config.dropout,
            norm="batch" if config.use_batch_norm else None,
            act=config.activation.lower(),
        )
        
        # 辅助UNet（处理下采样特征）
        aux_features = [f//2 for f in features]
        self.aux_unet = UNet(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.features[0],
            channels=aux_features,
            strides=strides,
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
        "UNet2D": UNet2DModel,
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
