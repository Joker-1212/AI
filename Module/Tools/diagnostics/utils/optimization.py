"""
性能优化工具

提供向量化计算和GPU加速的函数，用于优化计算密集型操作。
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union


def batch_rmse(pred: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    批量计算RMSE（向量化版本）
    
    参数:
        pred: 预测张量 (B, C, H, W) 或 (B, C, H, W, D)
        target: 目标张量，形状与pred相同
        reduction: 缩减方式，'mean'返回批量平均值，'none'返回每个样本的RMSE
        
    返回:
        RMSE值
    """
    diff = pred - target
    mse_per_sample = torch.mean(diff ** 2, dim=list(range(1, diff.dim())))
    rmse_per_sample = torch.sqrt(mse_per_sample)
    
    if reduction == 'mean':
        return rmse_per_sample.mean()
    elif reduction == 'none':
        return rmse_per_sample
    else:
        raise ValueError(f"不支持的reduction方式: {reduction}")


def batch_mae(pred: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    批量计算MAE（向量化版本）
    """
    diff = pred - target
    mae_per_sample = torch.mean(torch.abs(diff), dim=list(range(1, diff.dim())))
    
    if reduction == 'mean':
        return mae_per_sample.mean()
    elif reduction == 'none':
        return mae_per_sample
    else:
        raise ValueError(f"不支持的reduction方式: {reduction}")


def batch_psnr(pred: torch.Tensor, target: torch.Tensor, 
               data_range: Optional[float] = None, reduction: str = 'mean') -> torch.Tensor:
    """
    批量计算PSNR（向量化版本）
    """
    if data_range is None:
        data_range = target.max() - target.min()
        if data_range == 0:
            data_range = 1.0
    
    diff = pred - target
    mse_per_sample = torch.mean(diff ** 2, dim=list(range(1, diff.dim())))
    
    # 避免除零
    mse_per_sample = torch.clamp(mse_per_sample, min=1e-10)
    psnr_per_sample = 20 * torch.log10(data_range / torch.sqrt(mse_per_sample))
    
    if reduction == 'mean':
        return psnr_per_sample.mean()
    elif reduction == 'none':
        return psnr_per_sample
    else:
        raise ValueError(f"不支持的reduction方式: {reduction}")


def fast_ssim(pred: torch.Tensor, target: torch.Tensor, 
              data_range: float = 1.0, window_size: int = 11,
              size_average: bool = True) -> torch.Tensor:
    """
    快速SSIM计算（优化版本）
    
    使用分离卷积加速计算，支持批量处理。
    """
    device = pred.device
    channels = pred.size(1)
    
    # 创建高斯窗口
    from scipy.signal import gaussian
    gauss = gaussian(window_size, 1.5)
    gauss = torch.from_numpy(gauss).float().to(device)
    window = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    window = window / window.sum()
    window = window.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    
    # SSIM常数
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # 使用分离卷积计算均值
    def separable_conv2d(x, window):
        # 首先在高度维度卷积
        x = F.conv2d(x, window.unsqueeze(2), padding=(window_size//2, 0), groups=channels)
        # 然后在宽度维度卷积
        x = F.conv2d(x, window.unsqueeze(3), padding=(0, window_size//2), groups=channels)
        return x
    
    mu1 = separable_conv2d(pred, window)
    mu2 = separable_conv2d(target, window)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = separable_conv2d(pred * pred, window) - mu1_sq
    sigma2_sq = separable_conv2d(target * target, window) - mu2_sq
    sigma12 = separable_conv2d(pred * target, window) - mu1_mu2
    
    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


def normalize_batch(images: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    批量归一化图像到[0, 1]范围（GPU加速）
    """
    batch_min = images.view(images.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
    batch_max = images.view(images.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    
    # 扩展维度以匹配输入形状
    while batch_min.dim() < images.dim():
        batch_min = batch_min.unsqueeze(-1)
        batch_max = batch_max.unsqueeze(-1)
    
    range_val = batch_max - batch_min
    range_val = torch.where(range_val < eps, torch.ones_like(range_val), range_val)
    
    normalized = (images - batch_min) / range_val
    return normalized


def compute_metrics_batch_optimized(
    pred: torch.Tensor,
    target: torch.Tensor,
    metrics: List[str] = None,
    data_range: Optional[float] = None
) -> dict:
    """
    批量计算多个指标（优化版本）
    
    参数:
        pred: 预测张量 (B, C, H, W)
        target: 目标张量，形状相同
        metrics: 要计算的指标列表，支持['rmse', 'mae', 'psnr', 'ssim']
        data_range: 数据范围
        
    返回:
        指标字典
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'psnr', 'ssim']
    
    if data_range is None:
        data_range = float(target.max() - target.min())
        if data_range == 0:
            data_range = 1.0
    
    results = {}
    device = pred.device
    
    # 预计算常用值
    diff = pred - target
    
    if 'rmse' in metrics:
        rmse_per_sample = batch_rmse(pred, target, reduction='none')
        results['rmse'] = float(rmse_per_sample.mean().item())
        results['rmse_std'] = float(rmse_per_sample.std().item())
    
    if 'mae' in metrics:
        mae_per_sample = batch_mae(pred, target, reduction='none')
        results['mae'] = float(mae_per_sample.mean().item())
        results['mae_std'] = float(mae_per_sample.std().item())
    
    if 'psnr' in metrics:
        psnr_per_sample = batch_psnr(pred, target, data_range, reduction='none')
        results['psnr'] = float(psnr_per_sample.mean().item())
        results['psnr_std'] = float(psnr_per_sample.std().item())
    
    if 'ssim' in metrics:
        # 归一化用于SSIM计算
        pred_norm = normalize_batch(pred)
        target_norm = normalize_batch(target)
        
        # 批量计算SSIM（简化版本，实际应用中可能需要逐个样本计算）
        ssim_values = []
        batch_size = pred.size(0)
        for i in range(batch_size):
            ssim_val = fast_ssim(
                pred_norm[i:i+1], 
                target_norm[i:i+1], 
                data_range=1.0,
                size_average=True
            )
            ssim_values.append(ssim_val.item())
        
        results['ssim'] = float(np.mean(ssim_values))
        results['ssim_std'] = float(np.std(ssim_values))
    
    results['num_samples'] = pred.size(0)
    results['device'] = str(device)
    
    return results


def memory_efficient_gradient_check(
    model: torch.nn.Module,
    loss: torch.Tensor,
    max_layers: int = 10
) -> dict:
    """
    内存高效的梯度检查
    
    参数:
        model: PyTorch模型
        loss: 损失值
        max_layers: 最大检查层数
        
    返回:
        梯度统计
    """
    # 清零梯度
    model.zero_grad()
    
    # 计算梯度
    loss.backward()
    
    # 收集梯度统计
    gradient_stats = {
        'total_norm': 0.0,
        'layer_stats': {},
        'issues': []
    }
    
    layer_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            if layer_count >= max_layers:
                break
            
            grad = param.grad.data
            grad_norm = grad.norm().item()
            grad_mean = grad.mean().item()
            grad_std = grad.std().item()
            
            gradient_stats['total_norm'] += grad_norm ** 2
            
            # 检查问题
            issues = []
            if abs(grad_mean) < 1e-7 and grad_norm < 1e-5:
                issues.append('vanishing')
            if grad_norm > 1000.0:
                issues.append('exploding')
            if torch.any(torch.isnan(grad)):
                issues.append('nan')
            if torch.any(torch.isinf(grad)):
                issues.append('inf')
            
            gradient_stats['layer_stats'][name] = {
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'issues': issues
            }
            
            if issues:
                gradient_stats['issues'].append(f"{name}: {issues}")
            
            layer_count += 1
    
    gradient_stats['total_norm'] = np.sqrt(gradient_stats['total_norm'])
    
    return gradient_stats


__all__ = [
    'batch_rmse',
    'batch_mae',
    'batch_psnr',
    'fast_ssim',
    'normalize_batch',
    'compute_metrics_batch_optimized',
    'memory_efficient_gradient_check'
]
