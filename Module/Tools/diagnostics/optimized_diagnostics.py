"""
优化的诊断工具

提供性能优化的诊断计算功能，包括批量处理、GPU加速和内存优化。
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import warnings
from functools import lru_cache

from .utils.optimization import (
    batch_rmse, batch_mae, batch_psnr, fast_ssim, 
    compute_metrics_batch_optimized, normalize_batch
)
from ..memory_optimizer import MemoryMonitor, clear_memory
from ..device_manager import get_optimal_device


class OptimizedMetricsCalculator:
    """优化的指标计算器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化优化指标计算器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.device = get_optimal_device(
            memory_requirement_mb=self.config.get('memory_limit_mb', 1024),
            prefer_gpu=True
        )
        
        # 性能统计
        self.stats = {
            'computation_times': [],
            'memory_usage': [],
            'batch_sizes': []
        }
        
        # 启用缓存
        self.use_cache = self.config.get('use_cache', True)
        self.cache = {}
        
    def compute_all_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        metrics: List[str] = None,
        batch_size: Optional[int] = None,
        use_amp: bool = True
    ) -> Dict[str, Any]:
        """
        计算所有指标（优化版本）
        
        参数:
            pred: 预测张量
            target: 目标张量
            metrics: 要计算的指标列表
            batch_size: 批处理大小（None表示自动）
            use_amp: 是否使用混合精度
            
        返回:
            指标字典
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'psnr', 'ssim']
        
        start_time = time.time()
        
        # 检查缓存
        cache_key = self._create_cache_key(pred, target, metrics)
        if self.use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # 确保在正确的设备上
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        # 自动确定批处理大小
        if batch_size is None:
            batch_size = self._determine_optimal_batch_size(pred, target)
        
        results = {}
        
        # 分批处理大张量
        if pred.shape[0] > batch_size:
            results = self._compute_metrics_batched(
                pred, target, metrics, batch_size, use_amp
            )
        else:
            # 单批处理
            with torch.cuda.amp.autocast(enabled=use_amp and self.device.type == 'cuda'):
                results = compute_metrics_batch_optimized(
                    pred, target, metrics=metrics
                )
        
        # 记录性能统计
        computation_time = time.time() - start_time
        memory_usage = self._get_memory_usage()
        
        self.stats['computation_times'].append(computation_time)
        self.stats['memory_usage'].append(memory_usage)
        self.stats['batch_sizes'].append(pred.shape[0])
        
        # 缓存结果
        if self.use_cache:
            self.cache[cache_key] = results
        
        # 清理内存
        clear_memory(str(self.device))
        
        return results
    
    def _compute_metrics_batched(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        metrics: List[str],
        batch_size: int,
        use_amp: bool
    ) -> Dict[str, Any]:
        """分批计算指标"""
        num_batches = (pred.shape[0] + batch_size - 1) // batch_size
        
        all_results = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, pred.shape[0])
            
            pred_batch = pred[start_idx:end_idx]
            target_batch = target[start_idx:end_idx]
            
            with torch.cuda.amp.autocast(enabled=use_amp and self.device.type == 'cuda'):
                batch_results = compute_metrics_batch_optimized(
                    pred_batch, target_batch, metrics=metrics
                )
            
            all_results.append(batch_results)
            
            # 清理批次内存
            del pred_batch, target_batch
            if i % 10 == 0:  # 每10个批次清理一次
                clear_memory(str(self.device))
        
        # 合并结果
        merged_results = self._merge_batch_results(all_results, metrics)
        return merged_results
    
    def _merge_batch_results(
        self,
        batch_results: List[Dict[str, Any]],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """合并批次结果"""
        if not batch_results:
            return {}
        
        merged = {}
        
        for metric in metrics:
            if metric in batch_results[0]:
                # 收集所有批次的值
                values = [r[metric] for r in batch_results if metric in r]
                
                if values:
                    merged[metric] = float(np.mean(values))
                    
                    # 如果有标准差，合并它们
                    std_key = f"{metric}_std"
                    if std_key in batch_results[0]:
                        std_values = [r[std_key] for r in batch_results if std_key in r]
                        if std_values:
                            # 合并标准差
                            merged[std_key] = float(np.sqrt(np.mean(np.square(std_values))))
        
        # 添加样本数量
        merged['num_samples'] = sum(r.get('num_samples', 0) for r in batch_results)
        merged['device'] = str(self.device)
        
        return merged
    
    def _determine_optimal_batch_size(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> int:
        """确定最优批处理大小"""
        # 基于内存使用估计
        element_size = pred.element_size()  # 字节
        tensor_size = pred.numel() + target.numel()
        memory_per_sample = tensor_size * element_size / 1024 / 1024  # MB
        
        if self.device.type == 'cuda':
            # GPU内存
            free_memory = torch.cuda.mem_get_info(self.device)[0] / 1024 / 1024  # MB
            safe_memory = free_memory * 0.7  # 使用70%的可用内存
            batch_size = max(1, int(safe_memory / memory_per_sample))
        else:
            # CPU内存
            import psutil
            free_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
            safe_memory = free_memory * 0.5  # 使用50%的可用内存
            batch_size = max(1, int(safe_memory / memory_per_sample))
        
        # 限制范围
        batch_size = min(batch_size, 64)  # 最大64
        batch_size = max(batch_size, 1)   # 最小1
        
        return batch_size
    
    def _create_cache_key(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        metrics: List[str]
    ) -> str:
        """创建缓存键"""
        # 使用张量形状和指标创建简单的哈希键
        shape_hash = hash(pred.shape + target.shape)
        metrics_hash = hash(tuple(sorted(metrics)))
        return f"{shape_hash}_{metrics_hash}"
    
    def _get_memory_usage(self) -> float:
        """获取内存使用"""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device) / 1024 / 1024
        else:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.stats['computation_times']:
            return {}
        
        return {
            'avg_computation_time_ms': np.mean(self.stats['computation_times']) * 1000,
            'min_computation_time_ms': min(self.stats['computation_times']) * 1000,
            'max_computation_time_ms': max(self.stats['computation_times']) * 1000,
            'avg_memory_usage_mb': np.mean(self.stats['memory_usage']),
            'max_memory_usage_mb': max(self.stats['memory_usage']),
            'total_samples': sum(self.stats['batch_sizes']),
            'num_calls': len(self.stats['computation_times']),
        }
    
    def clear_cache(self):
        """清理缓存"""
        self.cache.clear()
        clear_memory(str(self.device))
    
    def reset_stats(self):
        """重置统计"""
        self.stats = {
            'computation_times': [],
            'memory_usage': [],
            'batch_sizes': []
        }


class ParallelDiagnostics:
    """并行诊断计算"""
    
    def __init__(self, num_workers: int = 4):
        """
        初始化并行诊断
        
        参数:
            num_workers: 工作线程数
        """
        self.num_workers = num_workers
        
    def compute_metrics_parallel(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        metrics: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        并行计算指标
        
        参数:
            predictions: 预测张量列表
            targets: 目标张量列表
            metrics: 指标列表
            
        返回:
            指标字典列表
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'psnr', 'ssim']
        
        # 简单实现：顺序处理
        # 在实际应用中可以使用multiprocessing或torch.multiprocessing
        results = []
        
        for pred, target in zip(predictions, targets):
            calculator = OptimizedMetricsCalculator()
            result = calculator.compute_all_metrics(pred, target, metrics)
            results.append(result)
        
        return results


class DiagnosticProfiler:
    """诊断性能分析器"""
    
    def __init__(self):
        self.profiles = {}
        
    def profile_function(
        self,
        func,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        分析函数性能
        
        参数:
            func: 要分析的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        返回:
            性能分析结果
        """
        import cProfile
        import pstats
        import io
        
        # 内存监控
        monitor = MemoryMonitor(device="cuda" if torch.cuda.is_available() else "cpu")
        monitor.start()
        
        # CPU性能分析
        pr = cProfile.Profile()
        pr.enable()
        
        # 执行函数
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        pr.disable()
        
        # 获取分析结果
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # 前20个最耗时的函数
        
        # 内存快照
        monitor.snapshot("end")
        
        # 收集结果
        profile_result = {
            'execution_time_s': execution_time,
            'profile_output': s.getvalue(),
            'memory_report': monitor.report(detailed=False),
            'memory_leak_detected': monitor.detect_leaks(),
        }
        
        # 记录分析
        func_name = func.__name__
        if func_name not in self.profiles:
            self.profiles[func_name] = []
        
        self.profiles[func_name].append(profile_result)
        
        return profile_result
    
    def get_function_stats(self, func_name: str) -> Dict[str, Any]:
        """获取函数统计"""
        if func_name not in self.profiles or not self.profiles[func_name]:
            return {}
        
        profiles = self.profiles[func_name]
        execution_times = [p['execution_time_s'] for p in profiles]
        
        return {
            'call_count': len(profiles),
            'avg_execution_time_s': np.mean(execution_times),
            'min_execution_time_s': min(execution_times),
            'max_execution_time_s': max(execution_times),
            'std_execution_time_s': np.std(execution_times),
            'total_execution_time_s': sum(execution_times),
        }


def optimize_diagnostic_pipeline(
    config: Dict[str, Any],
    enable_amp: bool = True,
    enable_batching: bool = True,
    cache_size: int = 100
) -> Dict[str, Any]:
    """
    优化诊断管道配置
    
    参数:
        config: 原始配置
        enable_amp: 是否启用混合精度
        enable_batching: 是否启用批处理
        cache_size: 缓存大小
        
    返回:
        优化后的配置
    """
    optimized_config = config.copy()
    
    # 优化计算设置
    optimized_config['computation'] = {
        'use_amp': enable_amp and torch.cuda.is_available(),
        'use_batching': enable_batching,
        'auto_batch_size': True,
        'cache_enabled': True,
        'cache_size': cache_size,
        'precision': 'mixed' if enable_amp else 'full',
    }
    
    # 优化内存设置
    optimized_config['memory'] = {
        'monitor_enabled': True,
        'auto_cleanup': True,
        'cleanup_interval': 10,  # 每10次计算清理一次
        'max_memory_mb': 4096,   # 最大内存限制
    }
    
    # 优化性能监控
    optimized_config['performance'] = {
        'profile_enabled': True,
        'log_interval': 100,     # 每100次计算记录一次
        'stats_aggregation': 'moving_average',
        'moving_average_window': 50,
    }
    
    return optimized_config


def benchmark_diagnostics(
    calculator: OptimizedMetricsCalculator,
    test_data: List[Tuple[torch.Tensor, torch.Tensor]],
    num_iterations: int = 10
) -> Dict[str, Any]:
    """
    基准测试诊断性能
    
    参数:
        calculator: 指标计算器
        test_data: 测试数据列表
        num_iterations: 迭代次数
        
    返回:
        基准测试结果
    """
    results = {
        'iteration_times': [],
        'memory_usage': [],
        'throughput': []
    }
    
    for i in range(num_iterations):
        iteration_times = []
        
        for pred, target in test_data:
            start_time = time.time()
            
            # 计算指标
            _ = calculator.compute_all_metrics(pred, target)
            
            iteration_time = time.time() - start_time
            iteration_times.append(iteration_time)
        
        # 记录迭代统计
        avg_time = np.mean(iteration_times) if iteration_times else 0
        results['iteration_times'].append(avg_time)
        
        # 计算吞吐量（样本/秒）
        if avg_time > 0:
            throughput = len(test_data) / avg_time
            results['throughput'].append(throughput)
        
        # 记录内存使用
        memory_usage = calculator._get_memory_usage()
        results['memory_usage'].append(memory_usage)
    
    # 计算最终统计
    final_stats = {
        'avg_iteration_time_s': np.mean(results['iteration_times']) if results['iteration_times'] else 0,
        'std_iteration_time_s': np.std(results['iteration_times']) if results['iteration_times'] else 0,
        'avg_throughput_samples_per_sec': np.mean(results['throughput']) if results['throughput'] else 0,
        'avg_memory_usage_mb': np.mean(results['memory_usage']) if results['memory_usage'] else 0,
        'max_memory_usage_mb': max(results['memory_usage']) if results['memory_usage'] else 0,
    }
    
    return final_stats


__all__ = [
    'OptimizedMetricsCalculator',
    'ParallelDiagnostics',
    'DiagnosticProfiler',
    'optimize_diagnostic_pipeline',
    'benchmark_diagnostics',
]
