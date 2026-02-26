"""
简单优化的数据加载器
不使用多线程/多进程，仅通过数据预加载和内存优化提高性能
"""

import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Any, Optional
import warnings


class SimplePreloadedDataset(Dataset):
    """简单预加载数据集 - 将所有数据加载到内存中"""
    
    def __init__(self, original_dataset: Dataset, max_samples: int = 1000):
        """
        参数:
            original_dataset: 原始数据集
            max_samples: 最大预加载样本数（避免内存溢出）
        """
        self.original_dataset = original_dataset
        self.max_samples = max_samples
        self.data = []
        self._preload_data()
    
    def _preload_data(self):
        """预加载数据到内存（单线程）"""
        print(f"开始预加载数据集 ({len(self.original_dataset)} 个样本，最多{self.max_samples}个)...")
        start_time = time.time()
        
        # 限制预加载数量以避免内存问题
        load_count = min(len(self.original_dataset), self.max_samples)
        
        for i in range(load_count):
            try:
                sample = self.original_dataset[i]
                self.data.append(sample)
                
                # 每100个样本显示进度
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    speed = (i + 1) / elapsed if elapsed > 0 else 0
                    print(f"  已加载 {i + 1}/{load_count} 个样本 ({speed:.1f} 样本/秒)")
                    
            except Exception as e:
                print(f"预加载样本 {i} 失败: {e}")
                # 添加占位符，后续动态加载
                self.data.append(None)
        
        # 如果数据集有更多样本，记录但不预加载
        if len(self.original_dataset) > load_count:
            print(f"注意: 数据集有 {len(self.original_dataset)} 个样本，但只预加载了 {load_count} 个")
            self.remaining_samples = len(self.original_dataset) - load_count
        else:
            self.remaining_samples = 0
        
        elapsed = time.time() - start_time
        print(f"预加载完成，耗时 {elapsed:.2f} 秒，加载了 {len(self.data)} 个样本")
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        """获取样本，如果已预加载则直接返回，否则动态加载"""
        if idx < len(self.data):
            if self.data[idx] is not None:
                return self.data[idx]
        
        # 动态加载
        return self.original_dataset[idx]


class MemoryOptimizedDataLoader:
    """内存优化的数据加载器"""
    
    def __init__(self, dataset: Dataset, batch_size: int = 8, shuffle: bool = True,
                 pin_memory: bool = True, preload: bool = True):
        """
        参数:
            dataset: 数据集
            batch_size: 批处理大小
            shuffle: 是否打乱数据
            pin_memory: 是否锁定内存（加速GPU传输）
            preload: 是否预加载数据
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory and torch.cuda.is_available()
        
        # 根据数据集大小决定是否预加载
        if preload and len(dataset) <= 1000:  # 小数据集才预加载
            print(f"数据集较小 ({len(dataset)} 个样本)，启用预加载...")
            self.dataset = SimplePreloadedDataset(dataset)
        else:
            self.dataset = dataset
            if preload:
                print(f"数据集较大 ({len(dataset)} 个样本)，跳过预加载以避免内存问题")
        
        # 创建标准DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # 不使用多进程
            pin_memory=self.pin_memory,
            prefetch_factor=None,
            persistent_workers=False
        )
        
        print(f"内存优化数据加载器已创建: batch_size={batch_size}, pin_memory={self.pin_memory}")
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def create_simple_optimized_dataloader(dataset: Dataset, batch_size: int = 8, 
                                       shuffle: bool = True, config: Optional[Any] = None) -> DataLoader:
    """
    创建简单优化的数据加载器
    
    参数:
        dataset: 数据集
        batch_size: 批处理大小
        shuffle: 是否打乱数据
        config: 配置对象（可选）
    
    返回:
        优化后的数据加载器
    """
    # 从配置中获取参数
    pin_memory = True
    preload = True
    
    if config is not None:
        pin_memory = getattr(config, 'pin_memory', True)
        # 根据数据集大小决定是否预加载
        dataset_size = len(dataset)
        if hasattr(config, 'max_preload_samples'):
            max_preload = getattr(config, 'max_preload_samples', 1000)
            preload = dataset_size <= max_preload
        else:
            preload = dataset_size <= 1000
    
    # 创建优化加载器
    return MemoryOptimizedDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        preload=preload
    ).dataloader


def benchmark_dataloader(dataloader: DataLoader, num_batches: int = 10) -> float:
    """
    基准测试数据加载器性能
    
    参数:
        dataloader: 数据加载器
        num_batches: 测试的批次数量
    
    返回:
        平均批次加载时间（秒）
    """
    print(f"开始基准测试 ({num_batches} 个批次)...")
    start_time = time.time()
    
    batch_times = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        batch_end = time.time()
        if i > 0:  # 跳过第一个批次（包含初始化时间）
            batch_times.append(batch_end - start_time)
        start_time = batch_end
    
    if batch_times:
        avg_time = np.mean(batch_times)
        print(f"基准测试完成: 平均批次加载时间 = {avg_time:.4f} 秒")
        return avg_time
    else:
        print("基准测试失败: 没有收集到有效的时间数据")
        return 0.0
