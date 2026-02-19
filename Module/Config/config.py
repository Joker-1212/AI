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
    image_size: Tuple[int, int, int] = (512, 512, 1)  # 图像尺寸 (H, W, D)
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
    loss_function: str = "L1Loss"  # 或 "MSELoss", "SSIMLoss", "MixedLoss", "MultiScaleLoss"
    loss_weights: Tuple[float, ...] = (1.0, 0.5, 0.1)  # 混合损失权重 [L1, SSIM, Perceptual]
    use_multi_scale_loss: bool = False  # 是否使用多尺度损失
    multi_scale_weights: Tuple[float, ...] = (1.0, 0.5, 0.25)  # 多尺度损失权重
    optimizer: str = "Adam"
    weight_decay: float = 1e-5
    scheduler: str = "ReduceLROnPlateau"
    patience: int = 10
    min_lr: float = 1e-6
    checkpoint_dir: str = "./models/checkpoints"
    log_dir: str = "./logs"
    device: str = "cuda"  # 或 "cpu"
    
    # 高级训练参数
    warmup_epochs: int = 0  # 学习率热身轮数
    gradient_clip_value: Optional[float] = None  # 梯度裁剪值
    gradient_clip_norm: Optional[float] = None  # 梯度裁剪范数
    use_early_stopping: bool = True  # 是否使用早停
    early_stopping_patience: int = 20  # 早停耐心值
    
    # 学习率调度器参数
    scheduler_factor: float = 0.5  # ReduceLROnPlateau的因子
    scheduler_step_size: int = 20  # StepLR的步长
    scheduler_gamma: float = 0.5  # StepLR的gamma
    scheduler_milestones: Tuple[int, ...] = (30, 60, 90)  # MultiStepLR的里程碑
    
    # 混合精度训练
    use_amp: bool = False  # 是否使用自动混合精度
    
    # 日志和监控
    log_interval: int = 10  # 日志记录间隔（批次）
    save_interval: int = 10  # 保存检查点间隔（轮数）
    monitor_metrics: Tuple[str, ...] = ("loss", "psnr", "ssim")  # 监控指标

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
