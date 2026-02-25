"""
工具函数
"""
import os
import time
import logging
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional
import yaml


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    获取配置好的日志记录器
    
    参数:
        name: 日志记录器名称
        level: 日志级别
        
    返回:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 如果已经有处理器，直接返回
    if logger.handlers:
        return logger
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                    epoch: int, val_loss: float, filepath: str,
                    additional_info: Optional[Dict[str, Any]] = None):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'timestamp': time.time() if 'time' in locals() else 0,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, filepath)
    print(f"检查点已保存到 {filepath}")


def load_checkpoint(filepath: str, model: nn.Module, 
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
    """加载检查点"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"检查点文件不存在: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('val_loss', float('inf'))
    
    print(f"从 epoch {epoch} 加载检查点，验证损失: {val_loss:.4f}")
    
    return epoch, val_loss


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor,
                     use_optimized: bool = True, **kwargs) -> Tuple[float, float]:
    """
    计算PSNR和SSIM指标
    
    参数:
        pred: 预测图像张量 (B, C, H, W, D)
        target: 目标图像张量 (B, C, H, W, D)
        use_optimized: 是否使用优化版本（推荐）
        **kwargs: 传递给优化函数的额外参数
    
    返回:
        psnr: 平均PSNR (dB)
        ssim: 平均SSIM
    """
    if use_optimized:
        try:
            from .memory_optimizer import calculate_metrics_optimized
            return calculate_metrics_optimized(pred, target, **kwargs)
        except ImportError:
            # 回退到原始实现
            print("警告: 无法导入优化版本，使用原始实现")
            pass
    
    # 原始实现（向后兼容）
    # 转换为numpy
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    batch_size = pred_np.shape[0]
    psnr_values = []
    ssim_values = []
    
    for i in range(batch_size):
        # 获取单张图像
        pred_img = pred_np[i, 0]  # 假设单通道
        target_img = target_np[i, 0]
        
        # 归一化到[0, 1]范围
        pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min() + 1e-8)
        target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min() + 1e-8)
        
        # 计算PSNR
        data_range = 1.0  # 因为我们已经归一化到[0,1]
        psnr = peak_signal_noise_ratio(target_img, pred_img, data_range=data_range)
        psnr_values.append(psnr)
        
        # 计算SSIM（对于3D图像，我们计算每个切片的平均值）
        if len(pred_img.shape) == 3:  # 3D图像
            slice_ssim = []
            for z in range(pred_img.shape[2]):
                pred_slice = pred_img[:, :, z]
                target_slice = target_img[:, :, z]
                ssim_val = structural_similarity(
                    target_slice, pred_slice,
                    data_range=data_range,
                    win_size=7  # 较小的窗口适用于256x256图像
                )
                slice_ssim.append(ssim_val)
            ssim_values.append(np.mean(slice_ssim))
        else:  # 2D图像
            ssim_val = structural_similarity(
                target_img, pred_img,
                data_range=data_range,
                win_size=7
            )
            ssim_values.append(ssim_val)
    
    return np.mean(psnr_values), np.mean(ssim_values)


def visualize_results(low_dose: torch.Tensor, enhanced: torch.Tensor, 
                      full_dose: torch.Tensor, save_path: Optional[str] = None):
    """
    可视化结果：低剂量、增强、全剂量图像对比
    
    参数:
        low_dose: 低剂量输入图像 (C, H, W, D)
        enhanced: 增强输出图像 (C, H, W, D)
        full_dose: 全剂量目标图像 (C, H, W, D)
        save_path: 保存路径（可选）
    """
    # 转换为numpy
    low_np = low_dose.detach().cpu().numpy()[0, 0]  # 取第一个通道
    enh_np = enhanced.detach().cpu().numpy()[0, 0]
    full_np = full_dose.detach().cpu().numpy()[0, 0]
    
    # 选择中间切片
    if len(low_np.shape) == 3:
        slice_idx = low_np.shape[2] // 2
        low_slice = low_np[:, :, slice_idx]
        enh_slice = enh_np[:, :, slice_idx]
        full_slice = full_np[:, :, slice_idx]
    else:
        low_slice = low_np
        enh_slice = enh_np
        full_slice = full_np
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(low_slice, cmap='gray')
    axes[0].set_title('低剂量 CT')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(enh_slice, cmap='gray')
    axes[1].set_title('增强 CT')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    im3 = axes[2].imshow(full_slice, cmap='gray')
    axes[2].set_title('全剂量 CT')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到 {save_path}")
    
    plt.show()


def save_config(config: Any, filepath: str):
    """保存配置到YAML文件"""
    with open(filepath, 'w') as f:
        yaml.dump(config.__dict__, f, default_flow_style=False)
    print(f"配置已保存到 {filepath}")


def load_config(filepath: str, config_class: Any) -> Any:
    """从YAML文件加载配置
    
    注意: 现在推荐使用 Config.from_yaml() 方法
    """
    import warnings
    warnings.warn(
        "load_config() is deprecated. Use Config.from_yaml() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # 尝试使用新的 from_yaml 方法
    if hasattr(config_class, 'from_yaml'):
        return config_class.from_yaml(filepath)
    
    # 回退到旧实现
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 递归创建配置对象
    def dict_to_config(d, cls):
        if hasattr(cls, '__dataclass_fields__'):
            # 这是一个数据类
            field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
            kwargs = {}
            for key, value in d.items():
                if key in field_types:
                    field_type = field_types[key]
                    # 检查是否是嵌套的数据类
                    if hasattr(field_type, '__dataclass_fields__'):
                        kwargs[key] = dict_to_config(value, field_type)
                    else:
                        kwargs[key] = value
            return cls(**kwargs)
        return d
    
    return dict_to_config(config_dict, config_class)


def ensure_dir(dir_path: str):
    """确保目录存在"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """将张量转换为图像numpy数组"""
    img = tensor.detach().cpu().numpy()
    if len(img.shape) == 4:  # (C, H, W, D)
        img = img[0, 0]  # 取第一个通道和第一个深度切片
    elif len(img.shape) == 3:  # (H, W, D)
        img = img[:, :, img.shape[2]//2]  # 取中间切片
    return img
