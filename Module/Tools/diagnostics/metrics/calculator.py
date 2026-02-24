"""
图像质量指标计算器

核心类：ImageMetricsCalculator
提供批量计算图像质量指标的功能，支持CPU和GPU加速。
"""

import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import torch
import torch.nn.functional as F

from . import (
    SKIMAGE_METRICS_AVAILABLE,
    LPIPS_AVAILABLE,
    mean_squared_error,
    mean_absolute_error,
    peak_signal_noise_ratio,
    structural_similarity,
    lpips_loss
)
from ..config import DiagnosticsConfig


class ImageMetricsCalculator:
    """
    图像质量指标计算器
    
    支持以下指标：
    - RMSE（均方根误差）
    - MAE（平均绝对误差）
    - PSNR（峰值信噪比）
    - SSIM（结构相似性）
    - MS-SSIM（多尺度SSIM）
    - LPIPS（学习感知图像块相似度）
    """
    
    def __init__(self, config: Optional[DiagnosticsConfig] = None):
        """
        初始化指标计算器
        
        参数:
            config: 诊断配置，如果为None则使用默认配置
        """
        self.config = config or DiagnosticsConfig()
        
    def calculate_all_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_range: Optional[float] = None
    ) -> Dict[str, float]:
        """
        计算所有启用的指标
        
        参数:
            pred: 预测图像张量 (B, C, H, W) 或 (B, C, H, W, D)
            target: 目标图像张量，与pred形状相同
            data_range: 图像数据范围，如果为None则自动计算
            
        返回:
            指标字典，键为指标名称，值为指标值
        """
        # 根据批处理大小选择计算方法
        batch_size = pred.shape[0]
        if batch_size > 1:
            return self.calculate_all_metrics_batch(pred, target, data_range, use_gpu=True)
        else:
            return self._calculate_single_sample_metrics(pred, target, data_range)
    
    def calculate_detailed_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_range: Optional[float] = None
    ) -> Dict[str, float]:
        """
        计算详细指标（calculate_all_metrics的别名，用于向后兼容）
        """
        return self.calculate_all_metrics(pred, target, data_range)
    
    def _get_spatial_dims(self, tensor: torch.Tensor) -> List[int]:
        """
        获取空间维度索引
        
        参数:
            tensor: 输入张量
            
        返回:
            空间维度索引列表
        """
        ndim = tensor.dim()
        if ndim == 4:  # (B, C, H, W)
            return [2, 3]
        elif ndim == 5:  # (B, C, H, W, D)
            return [2, 3, 4]
        else:
            raise ValueError(f"不支持的张量维度: {ndim}，期望4D或5D张量")
    
    def calculate_all_metrics_batch(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_range: Optional[float] = None,
        use_gpu: bool = True
    ) -> Dict[str, float]:
        """
        批量计算所有启用的指标（向量化版本，性能更优）
        
        支持4D张量 (B, C, H, W) 和5D张量 (B, C, H, W, D)
        """
        # 确保张量在相同设备上
        device = pred.device
        if use_gpu and torch.cuda.is_available() and device.type != 'cuda':
            pred = pred.cuda()
            target = target.cuda()
            device = pred.device
        
        # 检查张量维度
        if pred.dim() != target.dim():
            raise ValueError(f"预测和目标张量维度不匹配: pred={pred.dim()}D, target={target.dim()}D")
        
        ndim = pred.dim()
        if ndim not in [4, 5]:
            raise ValueError(f"不支持的张量维度: {ndim}，期望4D或5D张量")
        
        # 自动确定数据范围
        if data_range is None:
            data_range = float(max(target.max() - target.min(), 1.0))
        
        batch_size = pred.shape[0]
        metrics = {}
        
        # 向量化计算RMSE和MAE
        if self.config.compute_rmse or self.config.compute_mae:
            diff = pred - target
            
            # 根据维度确定要减少的维度
            if ndim == 4:
                reduce_dims = [1, 2, 3]  # 减少C, H, W维度
            else:  # ndim == 5
                reduce_dims = [1, 2, 3, 4]  # 减少C, H, W, D维度
            
            if self.config.compute_rmse:
                rmse_per_sample = torch.sqrt(torch.mean(diff ** 2, dim=reduce_dims))
                metrics['rmse'] = float(rmse_per_sample.mean().item())
                # 使用unbiased=False避免样本数为1时产生nan
                if rmse_per_sample.numel() > 1:
                    metrics['rmse_std'] = float(rmse_per_sample.std().item())
                else:
                    metrics['rmse_std'] = 0.0
            
            if self.config.compute_mae:
                mae_per_sample = torch.mean(torch.abs(diff), dim=reduce_dims)
                metrics['mae'] = float(mae_per_sample.mean().item())
                # 使用unbiased=False避免样本数为1时产生nan
                if mae_per_sample.numel() > 1:
                    metrics['mae_std'] = float(mae_per_sample.std().item())
                else:
                    metrics['mae_std'] = 0.0
        
        # 计算PSNR
        if self.config.compute_psnr:
            psnr_list = []
            for i in range(batch_size):
                pred_img = pred[i]
                target_img = target[i]
                mse = torch.mean((pred_img - target_img) ** 2)
                if mse == 0:
                    psnr = 100.0
                else:
                    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
                psnr_list.append(psnr.item())
            
            metrics['psnr'] = float(np.mean(psnr_list))
            # 使用ddof=0避免样本数为1时产生nan
            metrics['psnr_std'] = float(np.std(psnr_list, ddof=0))
        
        # 计算SSIM
        if self.config.compute_ssim:
            ssim_list = []
            for i in range(batch_size):
                pred_img = pred[i].unsqueeze(0)
                target_img = target[i].unsqueeze(0)
                
                pred_norm = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min() + 1e-8)
                target_norm = (target_img - target_img.min()) / (target_img.max() - target_img.min() + 1e-8)
                
                if ndim == 4:
                    ssim_val = self._calculate_ssim_torch(pred_norm, target_norm, data_range=1.0)
                else:  # ndim == 5
                    ssim_val = self._calculate_ssim_3d_torch(pred_norm, target_norm, data_range=1.0)
                
                ssim_list.append(ssim_val.item())
            
            metrics['ssim'] = float(np.mean(ssim_list))
            # 使用ddof=0避免样本数为1时产生nan
            metrics['ssim_std'] = float(np.std(ssim_list, ddof=0))
        
        # 计算LPIPS
        if self.config.compute_lpips and LPIPS_AVAILABLE:
            lpips_list = []
            for i in range(batch_size):
                pred_img = pred[i].unsqueeze(0)
                target_img = target[i].unsqueeze(0)
                
                pred_norm = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min() + 1e-8)
                target_norm = (target_img - target_img.min()) / (target_img.max() - target_img.min() + 1e-8)
                
                # LPIPS仅支持2D图像，对于3D图像我们计算每个切片的平均值
                if ndim == 5:
                    # 获取5D张量的形状 [B, C, H, W, D]
                    b, c, h, w, d = pred_norm.shape
                    
                    # 方法1：如果深度维度很小，计算每个深度切片的LPIPS
                    # 方法2：如果深度维度是空间维度之一，可能需要选择最大切片
                    # 这里我们采用简单的方法：如果深度<=4，计算所有切片平均值；否则使用中间切片
                    
                    if d <= 4:
                        # 深度较小，计算所有切片
                        slice_lpips = []
                        for depth_idx in range(d):
                            # 提取深度切片 [B, C, H, W, 1] -> [B, C, H, W]
                            pred_slice = pred_norm[:, :, :, :, depth_idx]
                            target_slice = target_norm[:, :, :, :, depth_idx]
                            slice_lpips.append(self._calculate_lpips(pred_slice, target_slice))
                        
                        lpips_val = np.mean(slice_lpips) if slice_lpips else 0.0
                    else:
                        # 深度较大，使用中间切片以避免计算开销
                        mid_depth = d // 2
                        pred_slice = pred_norm[:, :, :, :, mid_depth]
                        target_slice = target_norm[:, :, :, :, mid_depth]
                        lpips_val = self._calculate_lpips(pred_slice, target_slice)
                else:
                    lpips_val = self._calculate_lpips(pred_norm, target_norm)
                
                lpips_list.append(lpips_val)
            
            metrics['lpips'] = float(np.mean(lpips_list))
            metrics['lpips_std'] = float(np.std(lpips_list))
        
        metrics['num_samples'] = batch_size
        metrics['device'] = str(device)
        metrics['data_range'] = data_range
        metrics['input_dim'] = ndim
        
        return metrics
    
    def _calculate_single_sample_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_range: Optional[float] = None
    ) -> Dict[str, float]:
        """计算单个样本的指标"""
        metrics = {}
        
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        if data_range is None:
            data_range = float(max(target_np.max() - target_np.min(), 1.0))
        
        pred_norm = self._normalize(pred_np)
        target_norm = self._normalize(target_np)
        
        if self.config.compute_rmse:
            metrics['rmse'] = self._calculate_rmse(pred_np, target_np)
        
        if self.config.compute_mae:
            metrics['mae'] = self._calculate_mae(pred_np, target_np)
        
        if self.config.compute_psnr:
            metrics['psnr'] = self._calculate_psnr(pred_np, target_np, data_range)
        
        if self.config.compute_ssim:
            metrics['ssim'] = self._calculate_ssim(pred_norm, target_norm)
        
        if self.config.compute_lpips and LPIPS_AVAILABLE:
            lpips_val = self._calculate_lpips(
                torch.from_numpy(pred_norm).unsqueeze(0),
                torch.from_numpy(target_norm).unsqueeze(0)
            )
            metrics['lpips'] = lpips_val
        
        metrics['num_samples'] = 1
        metrics['data_range'] = data_range
        
        return metrics
    
    def _calculate_ssim_torch(self, pred: torch.Tensor, target: torch.Tensor,
                             data_range: float = 1.0) -> torch.Tensor:
        """PyTorch实现的SSIM计算（2D图像）"""
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        window_size = 11
        padding = window_size // 2
        
        mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=padding)
        mu2 = F.avg_pool2d(target, window_size, stride=1, padding=padding)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=padding) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=padding) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=padding) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def _calculate_ssim_3d_torch(self, pred: torch.Tensor, target: torch.Tensor,
                                data_range: float = 1.0) -> torch.Tensor:
        """PyTorch实现的SSIM计算（3D图像）"""
        # 获取图像尺寸
        _, _, depth, height, width = pred.shape
        
        # 检查图像是否太小无法进行3D SSIM计算
        min_dim = min(depth, height, width)
        window_size = 11
        
        if min_dim < window_size:
            # 图像太小，使用2D切片方法计算平均SSIM
            # 这通常发生在深度维度较小的情况下（如[1, 1, 256, 256, 3]）
            ssim_values = []
            
            # 遍历深度切片
            for d in range(depth):
                pred_slice = pred[:, :, d:d+1, :, :]  # 保持5D形状
                target_slice = target[:, :, d:d+1, :, :]
                
                # 移除深度维度，转换为2D图像
                pred_2d = pred_slice.squeeze(2)  # 形状: [1, 1, height, width]
                target_2d = target_slice.squeeze(2)
                
                # 计算2D SSIM
                ssim_val = self._calculate_ssim_torch(pred_2d, target_2d, data_range)
                ssim_values.append(ssim_val)
            
            # 返回平均值
            if ssim_values:
                return torch.stack(ssim_values).mean()
            else:
                return torch.tensor(0.0, device=pred.device)
        
        # 正常情况：图像足够大，使用3D SSIM计算
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        padding = window_size // 2
        
        # 使用3D平均池化
        mu1 = F.avg_pool3d(pred, window_size, stride=1, padding=padding)
        mu2 = F.avg_pool3d(target, window_size, stride=1, padding=padding)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool3d(pred * pred, window_size, stride=1, padding=padding) - mu1_sq
        sigma2_sq = F.avg_pool3d(target * target, window_size, stride=1, padding=padding) - mu2_sq
        sigma12 = F.avg_pool3d(pred * target, window_size, stride=1, padding=padding) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """将图像归一化到[0, 1]范围"""
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min > 1e-8:
            return (img - img_min) / (img_max - img_min)
        return img
    
    def _calculate_rmse(self, pred: np.ndarray, target: np.ndarray) -> float:
        """计算均方根误差"""
        return np.sqrt(mean_squared_error(target, pred))
    
    def _calculate_mae(self, pred: np.ndarray, target: np.ndarray) -> float:
        """计算平均绝对误差"""
        return mean_absolute_error(target, pred)
    
    def _calculate_psnr(self, pred: np.ndarray, target: np.ndarray, data_range: float) -> float:
        """计算峰值信噪比"""
        return peak_signal_noise_ratio(target, pred, data_range=data_range)
    
    def _calculate_ssim(self, pred: np.ndarray, target: np.ndarray) -> float:
        """计算结构相似性"""
        # 处理可能的批次和通道维度
        original_ndim = pred.ndim
        
        # 如果维度为5，可能是 (1, C, H, W, D) 或 (C, H, W, D, 1)
        if original_ndim == 5:
            # 移除大小为1的批次维度
            if pred.shape[0] == 1:
                pred = pred[0]
                target = target[0]
                original_ndim = 4
        
        # 如果维度为4，可能是 (C, H, W, D)
        if original_ndim == 4:
            # 计算每个深度切片的SSIM并取平均
            ssim_values = []
            for d in range(pred.shape[3]):
                pred_slice = pred[:, :, :, d]
                target_slice = target[:, :, :, d]
                ssim_val = self._calculate_ssim(pred_slice, target_slice)
                ssim_values.append(ssim_val)
            return float(np.mean(ssim_values)) if ssim_values else 0.0
        
        # 如果维度为3，可能是 (C, H, W) 或 (H, W, D)
        if original_ndim == 3:
            # 检查是2D+通道还是3D体数据
            if pred.shape[0] == 1:  # (1, H, W) - 单通道2D图像
                pred = pred[0]
                target = target[0]
                # 现在应该是2D
                return self._calculate_ssim(pred, target)
            elif pred.shape[2] <= 16:  # 假设深度维度较小，按切片处理
                ssim_values = []
                for z in range(pred.shape[2]):
                    pred_slice = pred[:, :, z]
                    target_slice = target[:, :, z]
                    ssim_val = self._calculate_ssim(pred_slice, target_slice)
                    ssim_values.append(ssim_val)
                return float(np.mean(ssim_values)) if ssim_values else 0.0
            else:
                # 可能是 (H, W, C) 格式，转换为 (C, H, W)
                pred = np.transpose(pred, (2, 0, 1))
                target = np.transpose(target, (2, 0, 1))
                return self._calculate_ssim(pred, target)
        
        # 如果维度为2，直接计算SSIM
        if original_ndim == 2:
            min_dim = min(pred.shape[0], pred.shape[1])
            win_size = min(7, min_dim)
            if win_size % 2 == 0:
                win_size = max(3, win_size - 1)
            if win_size < 3:
                return 0.0
            
            try:
                return structural_similarity(
                    target, pred,
                    data_range=1.0,
                    win_size=win_size,
                    channel_axis=None
                )
            except ValueError:
                win_size = min_dim
                if win_size % 2 == 0:
                    win_size = max(3, win_size - 1)
                return structural_similarity(
                    target, pred,
                    data_range=1.0,
                    win_size=win_size,
                    channel_axis=None
                )
        
        raise ValueError(f"不支持的图像维度: {original_ndim}")
    
    def _calculate_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """计算LPIPS指标"""
        if not LPIPS_AVAILABLE:
            return 0.0
        
        device = next(lpips_loss.parameters()).device
        pred = pred.to(device)
        target = target.to(device)
        
        with torch.no_grad():
            lpips_val = lpips_loss(pred, target, normalize=True)
        
        return lpips_val.item()
    
    def calculate_metric_distribution(
        self,
        preds: List[torch.Tensor],
        targets: List[torch.Tensor],
        metric_name: str = 'psnr'
    ) -> Dict[str, float]:
        """
        计算指标在多个样本上的分布
        """
        if len(preds) != len(targets):
            raise ValueError("预测和目标列表长度必须相同")
        
        metric_values = []
        
        for pred, target in zip(preds, targets):
            metrics = self.calculate_all_metrics(
                pred.unsqueeze(0) if pred.dim() == 3 else pred,
                target.unsqueeze(0) if target.dim() == 3 else target
            )
            
            if metric_name in metrics:
                metric_values.append(metrics[metric_name])
        
        if not metric_values:
            return {}
        
        metric_values = np.array(metric_values)
        
        return {
            'mean': float(np.mean(metric_values)),
            'std': float(np.std(metric_values, ddof=0)),  # 使用ddof=0避免样本数为1时产生nan
            'min': float(np.min(metric_values)),
            'max': float(np.max(metric_values)),
            'median': float(np.median(metric_values)),
            'q25': float(np.percentile(metric_values, 25)),
            'q75': float(np.percentile(metric_values, 75)),
            'num_samples': len(metric_values)
        }
