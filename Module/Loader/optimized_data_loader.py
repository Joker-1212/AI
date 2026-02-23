"""
优化的数据加载器

提供MONAI CacheDataset、PersistentDataset支持和流式数据处理优化。
"""

import os
import numpy as np
import psutil
from typing import Tuple, List, Dict, Optional, Any, Union
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from monai.data import (
    CacheDataset, 
    PersistentDataset,
    DataLoader as MonaiDataLoader,
    Dataset as MonaiDataset,
    decollate_batch
)
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensityRange,
    RandRotate,
    RandFlip,
    RandZoom,
    RandGaussianNoise,
    RandAdjustContrast,
    Resize,
    ToTensor,
    EnsureChannelFirstD,
    ScaleIntensityRangeD,
    RandRotateD,
    RandFlipD,
    RandZoomD,
    RandGaussianNoiseD,
    RandAdjustContrastD,
    ResizeD,
    ToTensorD,
    Transform,
    MapTransform,
    Randomizable,
    apply_transform
)
import nibabel as nib
from PIL import Image
import glob
import warnings
import gc
from pathlib import Path

from ..Config.config import DataConfig
from ..Tools.memory_optimizer import LargeImageProcessor, clear_memory


class StreamableCTDataset(Dataset):
    """
    流式CT数据集（内存优化）
    
    支持大图像的分块加载和流式处理。
    """
    
    def __init__(self, low_dose_paths: List[str], full_dose_paths: List[str], 
                 config: DataConfig, transform=None, is_train=True,
                 cache_size: int = 0, stream_chunk_size: Optional[int] = None):
        """
        参数:
            low_dose_paths: 低剂量CT图像路径列表
            full_dose_paths: 全剂量CT图像路径列表
            config: 数据配置
            transform: 可选的变换
            is_train: 是否为训练集
            cache_size: 缓存大小（0表示不缓存，>0表示内存缓存，-1表示磁盘缓存）
            stream_chunk_size: 流式处理块大小（None表示自动计算）
        """
        self.low_dose_paths = low_dose_paths
        self.full_dose_paths = full_dose_paths
        self.config = config
        self.transform = transform
        self.is_train = is_train
        self.cache_size = cache_size
        self.stream_chunk_size = stream_chunk_size
        
        # 确保路径匹配
        assert len(low_dose_paths) == len(full_dose_paths), \
            "低剂量和全剂量图像数量不匹配"
        
        # 初始化缓存
        self.cache = {}
        self.cache_order = []
        
        # 大图像处理器
        self.image_processor = LargeImageProcessor(
            max_memory_mb=config.max_memory_mb if hasattr(config, 'max_memory_mb') else 1024.0
        )
        
    def __len__(self):
        return len(self.low_dose_paths)
    
    def __getitem__(self, idx):
        # 检查缓存
        if idx in self.cache:
            return self.cache[idx]
        
        # 流式加载图像
        low_dose = self._stream_load_image(self.low_dose_paths[idx])
        full_dose = self._stream_load_image(self.full_dose_paths[idx])
        
        # 应用变换
        if self.transform:
            data = self.transform({"low": low_dose, "full": full_dose})
            low_dose = data["low"]
            full_dose = data["full"]
        
        # 缓存结果
        if self.cache_size != 0:
            self._add_to_cache(idx, (low_dose, full_dose))
        
        return low_dose, full_dose
    
    def _stream_load_image(self, path: str) -> np.ndarray:
        """流式加载图像（支持大图像分块加载）"""
        ext = os.path.splitext(path)[1].lower()
        
        if ext in ['.nii', '.nii.gz']:
            # NIfTI格式 - 使用分块加载
            return self._load_nifti_chunked(path)
        else:
            # 其他格式 - 直接加载
            return self._load_image_direct(path)
    
    def _load_nifti_chunked(self, path: str) -> np.ndarray:
        """分块加载NIfTI图像"""
        try:
            # 使用nibabel的延迟加载
            img = nib.load(path)
            data = img.get_fdata()
            
            # 如果图像太大，使用分块处理
            if data.nbytes > 100 * 1024 * 1024:  # 大于100MB
                warnings.warn(f"大图像检测: {path} ({data.nbytes/1024/1024:.1f} MB)")
                
                # 转换为张量进行分块处理
                tensor = torch.from_numpy(data.astype(np.float32))
                
                # 定义处理函数（这里只是简单的归一化）
                def process_chunk(chunk):
                    # 简单的归一化
                    chunk_min = chunk.min()
                    chunk_max = chunk.max()
                    chunk_range = chunk_max - chunk_min
                    if chunk_range > 0:
                        return (chunk - chunk_min) / chunk_range
                    return chunk
                
                # 分块处理
                processed_chunks = self.image_processor.process_in_chunks(
                    tensor, 
                    process_chunk,
                    chunk_dim=-1,  # 在深度维度分块
                    chunk_size=self.stream_chunk_size
                )
                
                # 合并结果
                result = torch.cat(processed_chunks, dim=-1).numpy()
                return result
            
            return data.astype(np.float32)
            
        except Exception as e:
            warnings.warn(f"加载NIfTI图像失败 {path}: {e}")
            # 回退到直接加载
            return self._load_image_direct(path)
    
    def _load_image_direct(self, path: str) -> np.ndarray:
        """直接加载图像"""
        ext = os.path.splitext(path)[1].lower()
        
        if ext in ['.nii', '.nii.gz']:
            img = nib.load(path)
            data = img.get_fdata()
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            img = Image.open(path).convert('L')
            data = np.array(img)
        elif ext in ['.dcm', '.dicom']:
            try:
                import pydicom
                ds = pydicom.dcmread(path)
                data = ds.pixel_array
            except ImportError:
                raise ImportError("请安装pydicom以读取DICOM文件")
        elif ext in ['.npy']:
            data = np.load(path)
        else:
            raise ValueError(f"不支持的图像格式: {ext}")
        
        return data.astype(np.float32)
    
    def _add_to_cache(self, idx: int, data: Tuple):
        """添加到缓存"""
        if self.cache_size > 0:  # 内存缓存
            if len(self.cache) >= self.cache_size:
                # 移除最旧的缓存项
                oldest_idx = self.cache_order.pop(0)
                del self.cache[oldest_idx]
            
            self.cache[idx] = data
            self.cache_order.append(idx)
        
        elif self.cache_size == -1:  # 磁盘缓存
            # 这里可以实现磁盘缓存
            pass
    
    def clear_cache(self):
        """清理缓存"""
        self.cache.clear()
        self.cache_order.clear()
        gc.collect()
        clear_memory()


class OptimizedCacheDataset(CacheDataset):
    """
    优化的CacheDataset
    
    添加内存监控和自动缓存管理。
    """
    
    def __init__(self, data: List[Dict], transform: Optional[Transform] = None,
                 cache_num: int = 0, cache_rate: float = 1.0,
                 num_workers: int = 0, progress: bool = True,
                 memory_limit_mb: float = 1024.0):
        """
        参数:
            data: 数据列表
            transform: 变换
            cache_num: 缓存数量
            cache_rate: 缓存比例
            num_workers: 工作线程数
            progress: 是否显示进度
            memory_limit_mb: 内存限制（MB）
        """
        super().__init__(
            data=data,
            transform=transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=progress
        )
        
        self.memory_limit_mb = memory_limit_mb
        self.memory_monitor = None
        
    def _load_cache_item(self, idx: int):
        """加载缓存项（重写以添加内存监控）"""
        if self.memory_monitor is None:
            from ..Tools.memory_optimizer import MemoryMonitor
            self.memory_monitor = MemoryMonitor(device="cpu")
            self.memory_monitor.start()
        
        # 检查内存使用
        memory_stats = self.memory_monitor.get_cpu_memory()
        if memory_stats['rss_mb'] > self.memory_limit_mb:
            warnings.warn(f"内存使用过高: {memory_stats['rss_mb']:.1f} MB > {self.memory_limit_mb} MB")
            # 清理部分缓存
            self._prune_cache()
        
        return super()._load_cache_item(idx)
    
    def _prune_cache(self):
        """修剪缓存（减少内存使用）"""
        if hasattr(self, '_cache') and self._cache:
            # 移除一半的缓存项
            keys_to_remove = list(self._cache.keys())[:len(self._cache)//2]
            for key in keys_to_remove:
                del self._cache[key]
            
            gc.collect()
            warnings.warn(f"已修剪缓存，剩余 {len(self._cache)} 项")


def get_optimized_transforms(config: DataConfig, is_train: bool = True) -> Compose:
    """
    获取优化的数据变换管道
    
    参数:
        config: 数据配置
        is_train: 是否为训练集
        
    返回:
        变换管道
    """
    transforms = []
    
    # 加载图像
    transforms.append(LoadImage(reader="NibabelReader", image_only=True))
    
    # 确保通道优先
    transforms.append(EnsureChannelFirst())
    
    # 强度归一化
    transforms.append(ScaleIntensityRange(
        a_min=config.intensity_range[0] if hasattr(config, 'intensity_range') else -1000,
        a_max=config.intensity_range[1] if hasattr(config, 'intensity_range') else 3000,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ))
    
    if is_train:
        # 数据增强
        if hasattr(config, 'augmentation') and config.augmentation:
            # 随机旋转
            if config.augmentation.get('rotation', True):
                transforms.append(RandRotate(
                    range_x=config.augmentation.get('rotation_range', 15),
                    prob=config.augmentation.get('rotation_prob', 0.5)
                ))
            
            # 随机翻转
            if config.augmentation.get('flip', True):
                transforms.append(RandFlip(
                    prob=config.augmentation.get('flip_prob', 0.5)
                ))
            
            # 随机缩放
            if config.augmentation.get('zoom', True):
                transforms.append(RandZoom(
                    min_zoom=config.augmentation.get('min_zoom', 0.9),
                    max_zoom=config.augmentation.get('max_zoom', 1.1),
                    prob=config.augmentation.get('zoom_prob', 0.5)
                ))
            
            # 随机高斯噪声
            if config.augmentation.get('noise', True):
                transforms.append(RandGaussianNoise(
                    std=config.augmentation.get('noise_std', 0.01),
                    prob=config.augmentation.get('noise_prob', 0.3)
                ))
    
    # 调整大小（如果需要）
    if hasattr(config, 'target_size') and config.target_size:
        transforms.append(Resize(
            spatial_size=config.target_size
        ))
    
    # 转换为张量
    transforms.append(ToTensor(dtype=torch.float32))
    
    return Compose(transforms)


def create_optimized_dataloader(
    low_dose_paths: List[str],
    full_dose_paths: List[str],
    config: DataConfig,
    batch_size: int = 4,
    is_train: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    cache_type: str = "memory",  # "memory", "disk", "none"
    cache_size: int = 100,
    stream_processing: bool = False,
    memory_limit_mb: float = 2048.0
) -> DataLoader:
    """
    创建优化的数据加载器
    
    参数:
        low_dose_paths: 低剂量图像路径
        full_dose_paths: 全剂量图像路径
        config: 数据配置
        batch_size: 批大小
        is_train: 是否为训练集
        num_workers: 工作线程数
        pin_memory: 是否固定内存
        cache_type: 缓存类型
        cache_size: 缓存大小
        stream_processing: 是否启用流式处理
        memory_limit_mb: 内存限制（MB）
        
    返回:
        数据加载器
    """
    # 准备数据
    data = []
    for low_path, full_path in zip(low_dose_paths, full_dose_paths):
        data.append({
            "low": low_path,
            "full": full_path
        })
    
    # 获取变换
    transforms = get_optimized_transforms(config, is_train)
    
    # 选择数据集类型
    if stream_processing:
        # 流式处理数据集
        dataset = StreamableCTDataset(
            low_dose_paths=low_dose_paths,
            full_dose_paths=full_dose_paths,
            config=config,
            transform=transforms,
            is_train=is_train,
            cache_size=cache_size if cache_type == "memory" else 0,
            stream_chunk_size=None  # 自动计算
        )
    
    elif cache_type == "memory":
        # 内存缓存数据集
        dataset = OptimizedCacheDataset(
            data=data,
            transform=transforms,
            cache_num=min(cache_size, len(data)),
            num_workers=num_workers,
            memory_limit_mb=memory_limit_mb
        )
    
    elif cache_type == "disk":
        # 磁盘缓存数据集
        dataset = PersistentDataset(
            data=data,
            transform=transforms,
            cache_dir=os.path.join(config.cache_dir, "persistent_cache") if hasattr(config, 'cache_dir') else "./cache",
        )
    
    else:
        # 普通数据集
        from .data_loader import CTDataset
        dataset = CTDataset(
            low_dose_paths=low_dose_paths,
            full_dose_paths=full_dose_paths,
            config=config,
            transform=transforms,
            is_train=is_train
        )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=is_train
    )
    
    return dataloader


class DataLoaderOptimizer:
    """数据加载器优化器"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.stats = {
            'load_times': [],
            'memory_usage': [],
            'throughput': []
        }
    
    def optimize_parameters(self, dataset_size: int, image_size: Tuple[int, ...]) -> Dict[str, Any]:
        """
        优化数据加载器参数
        
        参数:
            dataset_size: 数据集大小
            image_size: 图像大小
            
        返回:
            优化后的参数
        """
        # 计算图像内存占用
        image_memory_mb = np.prod(image_size) * 4 / 1024 / 1024  # 假设float32
        
        # 自动确定批大小
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            batch_size = max(1, int(gpu_memory_mb * 0.3 / image_memory_mb))  # 使用30%的GPU内存
            batch_size = min(batch_size, 32)  # 最大32
        else:
            cpu_memory_mb = psutil.virtual_memory().total / 1024 / 1024
            batch_size = max(1, int(cpu_memory_mb * 0.2 / image_memory_mb))  # 使用20%的CPU内存
            batch_size = min(batch_size, 16)  # 最大16
        
        # 确定工作线程数
        cpu_cores = os.cpu_count() or 4
        num_workers = min(cpu_cores - 1, 8)  # 最多8个工作线程
        
        # 确定缓存策略
        total_dataset_memory_mb = dataset_size * image_memory_mb
        if total_dataset_memory_mb < 1000:  # 小于1GB
            cache_type = "memory"
            cache_size = dataset_size
        elif total_dataset_memory_mb < 10000:  # 小于10GB
            cache_type = "disk"
            cache_size = 0
        else:
            cache_type = "none"
            cache_size = 0
        
        # 是否启用流式处理
        stream_processing = image_memory_mb > 100  # 单张图像大于100MB
        
        return {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'cache_type': cache_type,
            'cache_size': cache_size,
            'stream_processing': stream_processing,
            'pin_memory': torch.cuda.is_available(),
            'prefetch_factor': 2,
        }
    
    def update_stats(self, load_time: float, memory_usage: float, batch_size: int):
        """更新统计信息"""
        self.stats['load_times'].append(load_time)
        self.stats['memory_usage'].append(memory_usage)
        
        # 计算吞吐量（图像/秒）
        if load_time > 0:
            throughput = batch_size / load_time
            self.stats['throughput'].append(throughput)
    
    def get_recommendations(self) -> Dict[str, Any]:
        """获取优化建议"""
        if not self.stats['load_times']:
            return {}
        
        avg_load_time = np.mean(self.stats['load_times'])
        avg_memory = np.mean(self.stats['memory_usage'])
        avg_throughput = np.mean(self.stats['throughput']) if self.stats['throughput'] else 0
        
        recommendations = {
            'current_performance': {
                'avg_load_time_ms': avg_load_time * 1000,
                'avg_memory_mb': avg_memory,
                'avg_throughput_imgs_per_sec': avg_throughput,
            },
            'suggestions': []
        }
        
        # 生成建议
        if avg_load_time > 0.1:  # 加载时间大于100ms
            recommendations['suggestions'].append(
                "考虑启用缓存或增加缓存大小以减少加载时间"
            )
        
        if avg_memory > 1024:  # 内存使用大于1GB
            recommendations['suggestions'].append(
                "考虑启用流式处理或减少缓存大小以降低内存使用"
            )
        
        if avg_throughput < 10:  # 吞吐量小于10图像/秒
            recommendations['suggestions'].append(
                "考虑增加num_workers或使用更快的存储设备"
            )
        
        return recommendations


def benchmark_dataloader(dataloader: DataLoader, num_batches: int = 10) -> Dict[str, Any]:
    """
    基准测试数据加载器性能
    
    参数:
        dataloader: 数据加载器
        num_batches: 测试的批次数量
        
    返回:
        性能指标
    """
    import time
    import psutil
    
    process = psutil.Process()
    
    metrics = {
        'batch_times': [],
        'memory_usage': [],
        'throughput': []
    }
    
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        batch_time = time.time()
        
        # 模拟处理（只是移动数据）
        if isinstance(batch, (list, tuple)):
            data = batch[0]
        else:
            data = batch
        
        # 记录内存使用
        memory_mb = process.memory_info().rss / 1024 / 1024
        metrics['memory_usage'].append(memory_mb)
        
        # 计算批处理时间
        if i > 0:  # 跳过第一个批次（包含初始化时间）
            metrics['batch_times'].append(time.time() - batch_time)
        
        # 清理
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    
    # 计算指标
    if metrics['batch_times']:
        metrics['avg_batch_time_ms'] = np.mean(metrics['batch_times']) * 1000
        metrics['std_batch_time_ms'] = np.std(metrics['batch_times']) * 1000
        metrics['throughput_imgs_per_sec'] = dataloader.batch_size / np.mean(metrics['batch_times'])
    else:
        metrics['avg_batch_time_ms'] = 0
        metrics['std_batch_time_ms'] = 0
        metrics['throughput_imgs_per_sec'] = 0
    
    metrics['avg_memory_mb'] = np.mean(metrics['memory_usage']) if metrics['memory_usage'] else 0
    metrics['max_memory_mb'] = max(metrics['memory_usage']) if metrics['memory_usage'] else 0
    metrics['total_time_s'] = total_time
    
    return metrics


__all__ = [
    'StreamableCTDataset',
    'OptimizedCacheDataset',
    'get_optimized_transforms',
    'create_optimized_dataloader',
    'DataLoaderOptimizer',
    'benchmark_dataloader',
]
