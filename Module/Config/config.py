"""
配置参数
"""
from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "./data"
    low_dose_dir: str = "qd"          # 低剂量CT图像目录
    full_dose_dir: str = "fd"        # 全剂量CT图像目录
    image_size: Tuple[int, int, int] = (256, 256, 1)  # 图像尺寸 (H, W, D)
    batch_size: int = 4
    num_workers: int = 0  # Windows 上设置为 0 以避免多进程问题
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    normalize: bool = True
    normalize_range: Tuple[float, float] = (-1000, 1000)  # CT值的HU范围

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "UNet2D"
    in_channels: int = 1
    out_channels: int = 1
    features: Tuple[int, ...] = (32, 64, 128, 256)  # 特征通道数
    dropout: float = 0.1
    use_batch_norm: bool = True
    activation: str = "ReLU"

@dataclass
class TrainingConfig:
    """训练配置"""
    learning_rate: float = 1e-4
    num_epochs: int = 100
    loss_function: str = "L1Loss"  # 或 "MSELoss", "SSIMLoss"
    optimizer: str = "Adam"
    weight_decay: float = 1e-5
    scheduler: str = "ReduceLROnPlateau"
    patience: int = 10
    min_lr: float = 1e-6
    checkpoint_dir: str = "./models/checkpoints"
    log_dir: str = "./logs"
    device: str = "cuda"  # 或 "cpu"

@dataclass
class Config:
    """总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        # 确保分割比例总和为1
        total = self.data.train_split + self.data.val_split + self.data.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"分割比例总和应为1，当前为{total}")
