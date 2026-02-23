"""
内存优化工具

提供内存泄漏检测、监控和优化功能，特别针对大尺寸医学图像处理。
"""

import torch
import gc
import psutil
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import time
import warnings
from contextlib import contextmanager


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, device: str = "cuda"):
        """
        初始化内存监控器
        
        参数:
            device: 监控的设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.snapshots = []
        self.start_time = None
        
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self.snapshots = []
        self.snapshot("start")
        
    def snapshot(self, label: str = ""):
        """记录内存快照"""
        snapshot = {
            'label': label,
            'timestamp': time.time() - (self.start_time if self.start_time else 0),
            'cpu_memory': self.get_cpu_memory(),
            'gpu_memory': self.get_gpu_memory() if self.device == "cuda" else None,
            'objects': self.count_objects(),
        }
        self.snapshots.append(snapshot)
        return snapshot
        
    def get_cpu_memory(self) -> Dict[str, float]:
        """获取CPU内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
        }
        
    def get_gpu_memory(self) -> Optional[Dict[str, float]]:
        """获取GPU内存使用情况"""
        if not torch.cuda.is_available():
            return None
            
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'max_allocated_mb': max_allocated,
            'cached_mb': reserved - allocated,
        }
        
    def count_objects(self) -> Dict[str, int]:
        """统计Python对象数量"""
        counts = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            counts[obj_type] = counts.get(obj_type, 0) + 1
        return counts
        
    def report(self, detailed: bool = False) -> str:
        """生成监控报告"""
        if not self.snapshots:
            return "没有可用的快照数据"
            
        report_lines = ["内存监控报告:"]
        report_lines.append(f"设备: {self.device}")
        report_lines.append(f"快照数量: {len(self.snapshots)}")
        
        if len(self.snapshots) >= 2:
            first = self.snapshots[0]
            last = self.snapshots[-1]
            
            # CPU内存变化
            cpu_diff = last['cpu_memory']['rss_mb'] - first['cpu_memory']['rss_mb']
            report_lines.append(f"CPU内存变化: {cpu_diff:+.2f} MB")
            
            # GPU内存变化
            if self.device == "cuda" and last['gpu_memory'] and first['gpu_memory']:
                gpu_diff = last['gpu_memory']['allocated_mb'] - first['gpu_memory']['allocated_mb']
                report_lines.append(f"GPU内存变化: {gpu_diff:+.2f} MB")
                
        if detailed:
            for i, snap in enumerate(self.snapshots):
                report_lines.append(f"\n快照 {i} ({snap['label']}):")
                report_lines.append(f"  时间: {snap['timestamp']:.2f}s")
                report_lines.append(f"  CPU内存: {snap['cpu_memory']['rss_mb']:.2f} MB")
                if snap['gpu_memory']:
                    report_lines.append(f"  GPU内存: {snap['gpu_memory']['allocated_mb']:.2f} MB")
                    
        return "\n".join(report_lines)
        
    def detect_leaks(self, threshold_mb: float = 100.0) -> bool:
        """检测内存泄漏"""
        if len(self.snapshots) < 2:
            return False
            
        first = self.snapshots[0]
        last = self.snapshots[-1]
        
        cpu_leak = last['cpu_memory']['rss_mb'] - first['cpu_memory']['rss_mb'] > threshold_mb
        
        gpu_leak = False
        if self.device == "cuda" and last['gpu_memory'] and first['gpu_memory']:
            gpu_leak = last['gpu_memory']['allocated_mb'] - first['gpu_memory']['allocated_mb'] > threshold_mb
            
        return cpu_leak or gpu_leak


def calculate_metrics_optimized(
    pred: torch.Tensor, 
    target: torch.Tensor,
    use_gpu: bool = True,
    batch_size: Optional[int] = None
) -> Tuple[float, float]:
    """
    优化的指标计算函数（修复内存泄漏）
    
    参数:
        pred: 预测图像张量 (B, C, H, W, D)
        target: 目标图像张量 (B, C, H, W, D)
        use_gpu: 是否使用GPU加速
        batch_size: 批处理大小（用于大图像）
        
    返回:
        psnr: 平均PSNR (dB)
        ssim: 平均SSIM
    """
    device = pred.device if use_gpu and torch.cuda.is_available() else torch.device("cpu")
    
    # 如果指定了批处理大小，分批处理大图像
    if batch_size is not None and pred.shape[0] > batch_size:
        return _calculate_metrics_batched(pred, target, batch_size, device)
    
    # 使用GPU加速的向量化计算
    if use_gpu and torch.cuda.is_available():
        return _calculate_metrics_gpu(pred, target)
    else:
        return _calculate_metrics_cpu(pred, target)


def _calculate_metrics_gpu(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """GPU加速的指标计算"""
    # 确保在GPU上
    if pred.device.type != 'cuda':
        pred = pred.cuda()
    if target.device.type != 'cuda':
        target = target.cuda()
    
    # 归一化到[0, 1]范围
    pred_min = pred.view(pred.shape[0], -1).min(dim=1)[0].view(-1, 1, 1, 1, 1)
    pred_max = pred.view(pred.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1, 1)
    target_min = target.view(target.shape[0], -1).min(dim=1)[0].view(-1, 1, 1, 1, 1)
    target_max = target.view(target.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1, 1)
    
    pred_range = pred_max - pred_min
    target_range = target_max - target_min
    
    # 避免除零
    pred_range = torch.where(pred_range < 1e-8, torch.ones_like(pred_range), pred_range)
    target_range = torch.where(target_range < 1e-8, torch.ones_like(target_range), target_range)
    
    pred_norm = (pred - pred_min) / pred_range
    target_norm = (target - target_min) / target_range
    
    # 计算PSNR
    mse = torch.mean((pred_norm - target_norm) ** 2, dim=[1, 2, 3, 4])
    mse = torch.clamp(mse, min=1e-10)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    psnr_mean = torch.mean(psnr).item()
    
    # 计算SSIM（简化版本，实际应用中可能需要更复杂的实现）
    # 这里使用2D SSIM的近似计算
    ssim_mean = _calculate_ssim_approximate(pred_norm, target_norm)
    
    # 清理中间张量
    del pred_norm, target_norm, mse, psnr
    torch.cuda.empty_cache()
    
    return float(psnr_mean), float(ssim_mean)


def _calculate_metrics_cpu(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """CPU上的指标计算（内存优化版本）"""
    # 转换为numpy，但分批处理以避免内存峰值
    batch_size = pred.shape[0]
    psnr_values = []
    ssim_values = []
    
    for i in range(batch_size):
        # 处理单张图像
        pred_img = pred[i].detach().cpu().numpy()
        target_img = target[i].detach().cpu().numpy()
        
        # 归一化
        pred_min, pred_max = pred_img.min(), pred_img.max()
        target_min, target_max = target_img.min(), target_img.max()
        
        pred_range = pred_max - pred_min if pred_max - pred_min > 1e-8 else 1.0
        target_range = target_max - target_min if target_max - target_min > 1e-8 else 1.0
        
        pred_norm = (pred_img - pred_min) / pred_range
        target_norm = (target_img - target_min) / target_range
        
        # 计算PSNR
        mse = np.mean((pred_norm - target_norm) ** 2)
        if mse < 1e-10:
            psnr = 100.0  # 极高PSNR
        else:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        psnr_values.append(psnr)
        
        # 计算SSIM（简化版本）
        ssim = _calculate_ssim_simple(pred_norm, target_norm)
        ssim_values.append(ssim)
        
        # 及时清理
        del pred_img, target_img, pred_norm, target_norm
        gc.collect()
    
    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


def _calculate_metrics_batched(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    batch_size: int,
    device: torch.device
) -> Tuple[float, float]:
    """分批处理大图像"""
    total_batches = (pred.shape[0] + batch_size - 1) // batch_size
    psnr_values = []
    ssim_values = []
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, pred.shape[0])
        
        pred_batch = pred[start_idx:end_idx]
        target_batch = target[start_idx:end_idx]
        
        psnr_batch, ssim_batch = calculate_metrics_optimized(
            pred_batch, target_batch, 
            use_gpu=(device.type == 'cuda'),
            batch_size=None  # 不再进一步分批
        )
        
        psnr_values.append(psnr_batch)
        ssim_values.append(ssim_batch)
        
        # 清理批次
        del pred_batch, target_batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


def _calculate_ssim_approximate(pred: torch.Tensor, target: torch.Tensor) -> float:
    """简化的SSIM近似计算"""
    # 对于3D图像，计算每个切片的平均值
    if len(pred.shape) == 5:  # (B, C, H, W, D)
        batch_size = pred.shape[0]
        depth = pred.shape[4]
        ssim_values = []
        
        for b in range(batch_size):
            batch_ssim = []
            for d in range(depth):
                pred_slice = pred[b, 0, :, :, d]
                target_slice = target[b, 0, :, :, d]
                
                # 计算均值和方差
                mu1 = torch.mean(pred_slice)
                mu2 = torch.mean(target_slice)
                sigma1 = torch.var(pred_slice)
                sigma2 = torch.var(target_slice)
                sigma12 = torch.mean((pred_slice - mu1) * (target_slice - mu2))
                
                # SSIM公式
                C1 = (0.01 * 1.0) ** 2
                C2 = (0.03 * 1.0) ** 2
                
                ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                          ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
                batch_ssim.append(ssim_val.item())
            
            ssim_values.append(np.mean(batch_ssim))
        
        return float(np.mean(ssim_values))
    else:
        # 2D图像
        mu1 = torch.mean(pred)
        mu2 = torch.mean(target)
        sigma1 = torch.var(pred)
        sigma2 = torch.var(target)
        sigma12 = torch.mean((pred - mu1) * (target - mu2))
        
        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2
        
        ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                  ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
        
        return float(ssim_val.item())


def _calculate_ssim_simple(pred: np.ndarray, target: np.ndarray) -> float:
    """简化的SSIM计算（CPU版本）"""
    mu1 = np.mean(pred)
    mu2 = np.mean(target)
    sigma1 = np.var(pred)
    sigma2 = np.var(target)
    sigma12 = np.mean((pred - mu1) * (target - mu2))
    
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    
    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
              ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
    
    return float(ssim_val)


@contextmanager
def memory_context(device: str = "cuda", threshold_mb: float = 100.0):
    """
    内存监控上下文管理器
    
    用法:
        with memory_context() as monitor:
            # 执行内存密集型操作
            result = heavy_computation()
            
        if monitor.detect_leaks():
            print("检测到内存泄漏!")
    """
    monitor = MemoryMonitor(device)
    monitor.start()
    
    try:
        yield monitor
    finally:
        monitor.snapshot("end")
        
        # 强制垃圾回收
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def optimize_tensor_memory(tensor: torch.Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """
    优化张量内存使用
    
    参数:
        tensor: 输入张量
        dtype: 目标数据类型
        
    返回:
        优化后的张量
    """
    if tensor.dtype == dtype:
        return tensor
    
    # 检查是否支持该数据类型
    if dtype == torch.float16 and not tensor.is_cuda:
        warnings.warn("CPU上的float16可能不会节省内存，保持原类型")
        return tensor
    
    return tensor.to(dtype)


def clear_memory(device: str = "cuda"):
    """
    清理内存
    
    参数:
        device: 清理的设备
    """
    gc.collect()
    
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


class LargeImageProcessor:
    """大图像处理器（内存优化）"""
    
    def __init__(self, max_memory_mb: float = 1024.0, device: str = "cuda"):
        """
        初始化大图像处理器
        
        参数:
            max_memory_mb: 最大内存限制（MB）
            device: 处理设备
        """
        self.max_memory_mb = max_memory_mb
        self.device = device
        
    def _calculate_chunk_size(self, image: torch.Tensor) -> int:
        """根据内存限制计算块大小"""
        # 估计单个体素的内存使用（字节）
        bytes_per_voxel = image.element_size()
        total_voxels = image.numel()
        total_memory_mb = (total_voxels * bytes_per_voxel) / 1024 / 1024
        
        if total_memory_mb <= self.max_memory_mb:
            return total_voxels  # 可以一次性处理
        
        # 计算块大小
        chunk_ratio = self.max_memory_mb / total_memory_mb
        chunk_voxels = int(total_voxels * chunk_ratio)
        
        # 确保至少处理一个切片
        if len(image.shape) >= 3:
            slice_voxels = image.shape[-3] * image.shape[-2]
            chunk_voxels = max(chunk_voxels, slice_voxels)
        
        return chunk_voxels
        
    def process_in_chunks(self, image: torch.Tensor, chunk_func, chunk_dim: int = -1,
                         chunk_size: Optional[int] = None):
        """
        分块处理大图像
        
        参数:
            image: 输入图像张量
            chunk_func: 处理每个块的函数，接受一个张量块并返回处理结果
            chunk_dim: 分块的维度
            chunk_size: 块大小（自动计算如果为None）
            
        返回:
            处理结果列表
        """
        if chunk_size is None:
            # 根据内存限制自动计算块大小
            chunk_size = self._calculate_chunk_size(image)
        
        # 获取图像维度
        total_size = image.shape[chunk_dim]
        num_chunks = (total_size + chunk_size - 1) // chunk_size
        
        results = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_size)
            
            # 提取块
            slice_obj = [slice(None)] * len(image.shape)
            slice_obj[chunk_dim] = slice(start_idx, end_idx)
            chunk = image[tuple(slice_obj)]
            
            # 处理块
            result = chunk_func(chunk)
            results.append(result)
            
            # 清理内存
            del chunk
            clear_memory(self.device)
        
        return results
        
    def process_3d_image(self, image: torch.Tensor, process_func,
                        slice_dim: int = 2) -> torch.Tensor:
        """
        分片处理3D图像
        
        参数:
            image: 3D图像张量 (C, H, W, D)
            process_func: 处理函数
            slice_dim: 切片维度
            
        返回:
            处理后的3D图像
        """
        if len(image.shape) != 4:
            raise ValueError(f"期望4D张量 (C, H, W, D)，但得到形状 {image.shape}")
        
        depth = image.shape[slice_dim]
        processed_slices = []
        
        for d in range(depth):
            # 提取切片
            if slice_dim == 0:
                slice_tensor = image[d:d+1]
            elif slice_dim == 1:
                slice_tensor = image[:, d:d+1]
            elif slice_dim == 2:
                slice_tensor = image[:, :, d:d+1]
            elif slice_dim == 3:
                slice_tensor = image[:, :, :, d:d+1]
            else:
                raise ValueError(f"不支持的切片维度: {slice_dim}")
            
            # 处理切片
            processed_slice = process_func(slice_tensor)
            processed_slices.append(processed_slice)
            
            # 清理内存
            del slice_tensor
            if d % 10 == 0:  # 每10个切片清理一次
                clear_memory(self.device)
        
        # 合并切片
        return torch.cat(processed_slices, dim=slice_dim)
