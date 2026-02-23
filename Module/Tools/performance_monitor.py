"""
性能监控和基准测试工具

提供全面的性能监控、基准测试和优化建议功能。
"""

import torch
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime
import json
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum


class MetricType(Enum):
    """指标类型"""
    TRAINING = "training"
    INFERENCE = "inference"
    MEMORY = "memory"
    GPU = "gpu"
    DATA_LOADING = "data_loading"
    DIAGNOSTICS = "diagnostics"


@dataclass
class PerformanceMetric:
    """性能指标"""
    name: str
    value: float
    unit: str
    timestamp: float
    metric_type: MetricType
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['metric_type'] = self.metric_type.value
        return result


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化性能监控器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.metrics: List[PerformanceMetric] = []
        self.snapshots: List[Dict[str, Any]] = []
        
        # 监控设置
        self.monitor_interval = self.config.get('monitor_interval', 1.0)  # 秒
        self.last_monitor_time = 0
        
        # 性能基准
        self.benchmarks = {}
        
    def start_monitoring(self):
        """开始监控"""
        self.metrics.clear()
        self.snapshots.clear()
        self.last_monitor_time = time.time()
        
    def record_metric(
        self,
        name: str,
        value: float,
        unit: str,
        metric_type: MetricType,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """记录指标"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            metric_type=metric_type,
            metadata=metadata or {}
        )
        self.metrics.append(metric)
        
        # 定期快照
        current_time = time.time()
        if current_time - self.last_monitor_time >= self.monitor_interval:
            self.take_snapshot()
            self.last_monitor_time = current_time
    
    def take_snapshot(self):
        """获取系统快照"""
        snapshot = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'cpu_count': psutil.cpu_count(),
            'memory': self._get_memory_info(),
            'gpu': self._get_gpu_info() if torch.cuda.is_available() else None,
            'process': self._get_process_info(),
        }
        self.snapshots.append(snapshot)
    
    def _get_memory_info(self) -> Dict[str, float]:
        """获取内存信息"""
        memory = psutil.virtual_memory()
        return {
            'total_mb': memory.total / 1024 / 1024,
            'available_mb': memory.available / 1024 / 1024,
            'used_mb': memory.used / 1024 / 1024,
            'percent': memory.percent,
        }
    
    def _get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """获取GPU信息"""
        if not torch.cuda.is_available():
            return None
        
        torch.cuda.synchronize()
        
        gpu_info = {
            'device_count': torch.cuda.device_count(),
            'devices': []
        }
        
        for i in range(torch.cuda.device_count()):
            device_info = {
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'memory_allocated_mb': torch.cuda.memory_allocated(i) / 1024 / 1024,
                'memory_reserved_mb': torch.cuda.memory_reserved(i) / 1024 / 1024,
                'max_memory_allocated_mb': torch.cuda.max_memory_allocated(i) / 1024 / 1024,
                'utilization': torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0,
            }
            gpu_info['devices'].append(device_info)
        
        return gpu_info
    
    def _get_process_info(self) -> Dict[str, Any]:
        """获取进程信息"""
        process = psutil.Process()
        
        return {
            'pid': process.pid,
            'memory_rss_mb': process.memory_info().rss / 1024 / 1024,
            'memory_vms_mb': process.memory_info().vms / 1024 / 1024,
            'cpu_percent': process.cpu_percent(interval=0.1),
            'num_threads': process.num_threads(),
            'create_time': process.create_time(),
        }
    
    def monitor_training(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        batch_size: int,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """监控训练性能"""
        # 记录训练指标
        self.record_metric(
            name="train_loss",
            value=train_loss,
            unit="",
            metric_type=MetricType.TRAINING,
            metadata={'epoch': epoch, 'batch_size': batch_size}
        )
        
        self.record_metric(
            name="val_loss",
            value=val_loss,
            unit="",
            metric_type=MetricType.TRAINING,
            metadata={'epoch': epoch}
        )
        
        self.record_metric(
            name="learning_rate",
            value=learning_rate,
            unit="",
            metric_type=MetricType.TRAINING,
            metadata={'epoch': epoch}
        )
        
        # 记录额外指标
        if additional_metrics:
            for name, value in additional_metrics.items():
                self.record_metric(
                    name=name,
                    value=value,
                    unit="",
                    metric_type=MetricType.TRAINING,
                    metadata={'epoch': epoch}
                )
    
    def monitor_inference(
        self,
        batch_size: int,
        inference_time: float,
        throughput: float,
        model_size_mb: float = None
    ):
        """监控推理性能"""
        self.record_metric(
            name="inference_time",
            value=inference_time,
            unit="seconds",
            metric_type=MetricType.INFERENCE,
            metadata={'batch_size': batch_size}
        )
        
        self.record_metric(
            name="throughput",
            value=throughput,
            unit="samples/second",
            metric_type=MetricType.INFERENCE,
            metadata={'batch_size': batch_size}
        )
        
        if model_size_mb is not None:
            self.record_metric(
                name="model_size",
                value=model_size_mb,
                unit="MB",
                metric_type=MetricType.INFERENCE
            )
    
    def monitor_memory(self):
        """监控内存使用"""
        # 系统内存
        memory = psutil.virtual_memory()
        self.record_metric(
            name="system_memory_used",
            value=memory.percent,
            unit="percent",
            metric_type=MetricType.MEMORY
        )
        
        # 进程内存
        process = psutil.Process()
        self.record_metric(
            name="process_memory_rss",
            value=process.memory_info().rss / 1024 / 1024,
            unit="MB",
            metric_type=MetricType.MEMORY
        )
        
        # GPU内存（如果可用）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            self.record_metric(
                name="gpu_memory_allocated",
                value=allocated,
                unit="MB",
                metric_type=MetricType.GPU
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if not self.metrics:
            return {}
        
        # 按指标类型分组
        metrics_by_type = {}
        for metric_type in MetricType:
            type_metrics = [m for m in self.metrics if m.metric_type == metric_type]
            if type_metrics:
                metrics_by_type[metric_type.value] = [
                    m.to_dict() for m in type_metrics
                ]
        
        # 计算统计信息
        stats = {}
        for metric_type in MetricType:
            type_metrics = [m for m in self.metrics if m.metric_type == metric_type]
            if type_metrics:
                values = [m.value for m in type_metrics]
                stats[metric_type.value] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values),
                }
        
        # 系统快照统计
        if self.snapshots:
            cpu_percents = [s['cpu_percent'] for s in self.snapshots]
            memory_percents = [s['memory']['percent'] for s in self.snapshots]
            
            stats['system'] = {
                'avg_cpu_percent': np.mean(cpu_percents),
                'max_cpu_percent': max(cpu_percents),
                'avg_memory_percent': np.mean(memory_percents),
                'max_memory_percent': max(memory_percents),
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_metrics': len(self.metrics),
            'monitoring_duration_s': self.metrics[-1].timestamp - self.metrics[0].timestamp if self.metrics else 0,
            'metrics_by_type': metrics_by_type,
            'statistics': stats,
            'snapshots_count': len(self.snapshots),
        }
    
    def save_report(self, filepath: str):
        """保存报告到文件"""
        report = self.get_performance_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"性能报告已保存到: {filepath}")
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """获取优化建议"""
        recommendations = []
        
        # 分析指标
        training_metrics = [m for m in self.metrics if m.metric_type == MetricType.TRAINING]
        memory_metrics = [m for m in self.metrics if m.metric_type == MetricType.MEMORY]
        gpu_metrics = [m for m in self.metrics if m.metric_type == MetricType.GPU]
        
        # 检查训练性能
        if training_metrics:
            train_losses = [m.value for m in training_metrics if m.name == "train_loss"]
            if train_losses and len(train_losses) > 10:
                # 检查训练损失是否收敛缓慢
                recent_losses = train_losses[-10:]
                loss_variance = np.var(recent_losses) / np.mean(recent_losses)
                if loss_variance > 0.1:
                    recommendations.append({
                        'category': 'training',
                        'issue': '训练损失波动较大',
                        'suggestion': '考虑降低学习率或增加批量大小',
                        'severity': 'medium'
                    })
        
        # 检查内存使用
        if memory_metrics:
            memory_values = [m.value for m in memory_metrics if m.name == "process_memory_rss"]
            if memory_values:
                avg_memory = np.mean(memory_values)
                if avg_memory > 4096:  # 大于4GB
                    recommendations.append({
                        'category': 'memory',
                        'issue': '内存使用过高',
                        'suggestion': '考虑启用流式处理或减少缓存大小',
                        'severity': 'high'
                    })
        
        # 检查GPU使用
        if gpu_metrics:
            gpu_memory = [m.value for m in gpu_metrics if m.name == "gpu_memory_allocated"]
            if gpu_memory:
                avg_gpu_memory = np.mean(gpu_memory)
                if avg_gpu_memory > 8000:  # 大于8GB
                    recommendations.append({
                        'category': 'gpu',
                        'issue': 'GPU内存使用过高',
                        'suggestion': '考虑使用混合精度训练或减少批量大小',
                        'severity': 'high'
                    })
        
        return recommendations


class BenchmarkSuite:
    """基准测试套件"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_training(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        num_epochs: int = 3,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        基准测试训练性能
        
        参数:
            model: 模型
            train_loader: 训练数据加载器
            num_epochs: 训练轮数
            device: 设备
            
        返回:
            基准测试结果
        """
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 简单训练循环
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        metrics = {
            'epoch_times': [],
            'batch_times': [],
            'memory_usage': [],
        }
        
        model.train()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 10:  # 只测试前10个批次
                    break
                
                batch_start = time.time()
                
                data = data.to(device)
                target = target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                batch_time = time.time() - batch_start
                metrics['batch_times'].append(batch_time)
                
                # 记录内存使用
                if device.type == 'cuda':
                    metrics['memory_usage'].append(
                        torch.cuda.memory_allocated(device) / 1024 / 1024
                    )
            
            epoch_time = time.time() - epoch_start
            metrics['epoch_times'].append(epoch_time)
        
        # 计算统计
        result = {
            'avg_epoch_time_s': np.mean(metrics['epoch_times']) if metrics['epoch_times'] else 0,
            'avg_batch_time_s': np.mean(metrics['batch_times']) if metrics['batch_times'] else 0,
            'throughput_samples_per_sec': train_loader.batch_size / np.mean(metrics['batch_times']) if metrics['batch_times'] else 0,
            'avg_gpu_memory_mb': np.mean(metrics['memory_usage']) if metrics['memory_usage'] else 0,
            'device': str(device),
        }
        
        self.results['training'] = result
        return result
    
    def benchmark_inference(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        基准测试推理性能
        
        参数:
            model: 模型
            input_shape: 输入形状
            num_iterations: 迭代次数
            warmup_iterations: 预热迭代次数
            device: 设备
            
        返回:
            基准测试结果
        """
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # 创建测试输入
        x = torch.randn(input_shape).to(device)
        
        metrics = {
            'inference_times': [],
            'memory_usage': [],
        }
        
        with torch.no_grad():
            for i in range(warmup_iterations + num_iterations):
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start_time = time.time()
                
                _ = model(x)
                
                torch.cuda.synchronize() if device.type == 'cuda' else None
                end_time = time.time()
                
                if i >= warmup_iterations:
                    inference_time = end_time - start_time
                    metrics['inference_times'].append(inference_time)
                    
                    # 记录内存使用
                    if device.type == 'cuda':
                        metrics['memory_usage'].append(
                            torch.cuda.memory_allocated(device) / 1024 / 1024
                        )
        
        # 计算统计
        result = {
            'avg_inference_time_ms': np.mean(metrics['inference_times']) * 1000 if metrics['inference_times'] else 0,
            'std_inference_time_ms': np.std(metrics['inference_times']) * 1000 if metrics['inference_times'] else 0,
            'throughput_samples_per_sec': 1 / np.mean(metrics['inference_times']) if metrics['inference_times'] else 0,
            'avg_gpu_memory_mb': np.mean(metrics['memory_usage']) if metrics['memory_usage'] else 0,
            'device': str(device),
        }
        
        self.results['inference'] = result
        return result
    
    def benchmark_data_loading(
        self,
        data_loader: torch.utils.data.DataLoader,
        num_batches: int = 20
    ) -> Dict[str, Any]:
        """
        基准测试数据加载性能
        
        参数:
            data_loader: 数据加载器
            num_batches: 测试的批次数量
            
        返回:
            基准测试结果
        """
        metrics = {
            'batch_load_times': [],
            'memory_usage': [],
        }
        
        process = psutil.Process()
        
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
            
            batch_start = time.time()
            
            # 模拟处理（只是访问数据）
            if isinstance(batch, (list, tuple)):
                data = batch[0]
            else:
                data = batch
            
            batch_time = time.time() - batch_start
            metrics['batch_load_times'].append(batch_time)
            
            # 记录内存使用
            metrics['memory_usage'].append(
                process.memory_info().rss / 1024 / 1024
            )
            
            # 清理
            del data
            gc.collect()
        
        # 计算统计
        result = {
            'avg_batch_load_time_ms': np.mean(metrics['batch_load_times']) * 1000 if metrics['batch_load_times'] else 0,
            'std_batch_load_time_ms': np.std(metrics['batch_load_times']) * 1000 if metrics['batch_load_times'] else 0,
            'throughput_samples_per_sec': data_loader.batch_size / np.mean(metrics['batch_load_times']) if metrics['batch_load_times'] else 0,
            'avg_memory_mb': np.mean(metrics['memory_usage']) if metrics['memory_usage'] else 0,
            'max_memory_mb': max(metrics['memory_usage']) if metrics['memory_usage'] else 0,
        }
        
        self.results['data_loading'] = result
        return result
    
    def run_comprehensive_benchmark(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_input_shape: Tuple[int, ...],
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        运行综合基准测试
        
        参数:
            model: 模型
            train_loader: 训练数据加载器
            test_input_shape: 测试输入形状
            device: 设备
            
        返回:
            综合基准测试结果
        """
        print("开始综合性能基准测试...")
        
        results = {}
        
        # 1. 数据加载基准测试
        print("1. 基准测试数据加载性能...")
        data_loading_result = self.benchmark_data_loading(train_loader)
        results['data_loading'] = data_loading_result
        
        # 2. 推理基准测试
        print("2. 基准测试推理性能...")
        inference_result = self.benchmark_inference(
            model, test_input_shape, device=device
        )
        results['inference'] = inference_result
        
        # 3. 训练基准测试（简化）
        print("3. 基准测试训练性能...")
        training_result = self.benchmark_training(
            model, train_loader, num_epochs=2, device=device
        )
        results['training'] = training_result
        
        # 4. 内存基准测试
        print("4. 基准测试内存使用...")
        memory_result = self._benchmark_memory(model, test_input_shape, device)
        results['memory'] = memory_result
        
        # 生成综合报告
        comprehensive_report = {
            'timestamp': datetime.now().isoformat(),
            'device': device,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024,
            'benchmarks': results,
            'summary': self._generate_summary(results),
        }
        
        self.results['comprehensive'] = comprehensive_report
        return comprehensive_report
    
    def _benchmark_memory(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """基准测试内存使用"""
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 测试前内存
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            before_memory = torch.cuda.memory_allocated(device) / 1024 / 1024
        else:
            process = psutil.Process()
            before_memory = process.memory_info().rss / 1024 / 1024
        
        # 前向传播
        x = torch.randn(input_shape).to(device)
        with torch.no_grad():
            _ = model(x)
        
        # 测试后内存
        if device.type == 'cuda':
            after_memory = torch.cuda.memory_allocated(device) / 1024 / 1024
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        else:
            process = psutil.Process()
            after_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = after_memory
        
        return {
            'before_memory_mb': before_memory,
            'after_memory_mb': after_memory,
            'peak_memory_mb': peak_memory,
            'memory_increase_mb': after_memory - before_memory,
            'device': str(device),
        }
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成基准测试摘要"""
        summary = {
            'performance_score': 0,
            'bottlenecks': [],
            'recommendations': [],
        }
        
        # 检查数据加载性能
        if 'data_loading' in results:
            dl_result = results['data_loading']
            if dl_result.get('avg_batch_load_time_ms', 0) > 100:  # 大于100ms
                summary['bottlenecks'].append('数据加载速度慢')
                summary['recommendations'].append('考虑启用数据缓存或增加num_workers')
        
        # 检查推理性能
        if 'inference' in results:
            inf_result = results['inference']
            if inf_result.get('avg_inference_time_ms', 0) > 50:  # 大于50ms
                summary['bottlenecks'].append('推理速度慢')
                summary['recommendations'].append('考虑使用混合精度或模型优化')
        
        # 检查内存使用
        if 'memory' in results:
            mem_result = results['memory']
            if mem_result.get('peak_memory_mb', 0) > 4096:  # 大于4GB
                summary['bottlenecks'].append('内存使用过高')
                summary['recommendations'].append('考虑减少批量大小或启用内存优化')
        
        # 计算性能分数（简单实现）
        performance_score = 100
        
        if summary['bottlenecks']:
            performance_score -= len(summary['bottlenecks']) * 20
        
        summary['performance_score'] = max(0, min(100, performance_score))
        
        return summary
    
    def save_benchmark_results(self, filepath: str):
        """保存基准测试结果"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"基准测试结果已保存到: {filepath}")


@contextmanager
def performance_context(name: str, monitor: Optional[PerformanceMonitor] = None):
    """
    性能监控上下文管理器
    
    用法:
        with performance_context("my_function", monitor) as ctx:
            result = my_function()
            ctx.record_metric("custom_metric", 42.0, "unit")
    """
    class PerformanceContext:
        def __init__(self, name, monitor):
            self.name = name
            self.monitor = monitor
            self.start_time = time.time()
        
        def record_metric(self, name, value, unit, metric_type=MetricType.DIAGNOSTICS):
            if self.monitor:
                self.monitor.record_metric(name, value, unit, metric_type)
        
        def get_elapsed_time(self):
            return time.time() - self.start_time
    
    context = PerformanceContext(name, monitor)
    
    try:
        yield context
    finally:
        # 记录执行时间
        elapsed_time = context.get_elapsed_time()
        if monitor:
            monitor.record_metric(
                name=f"{name}_execution_time",
                value=elapsed_time,
                unit="seconds",
                metric_type=MetricType.DIAGNOSTICS,
                metadata={'function_name': name}
            )


def create_performance_dashboard(
    monitor: PerformanceMonitor,
    benchmark_suite: BenchmarkSuite
) -> Dict[str, Any]:
    """
    创建性能仪表板
    
    参数:
        monitor: 性能监控器
        benchmark_suite: 基准测试套件
        
    返回:
        仪表板数据
    """
    dashboard = {
        'timestamp': datetime.now().isoformat(),
        'performance_report': monitor.get_performance_report(),
        'benchmark_results': benchmark_suite.results,
        'optimization_recommendations': monitor.get_optimization_recommendations(),
        'system_info': {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    }
    
    return dashboard


def optimize_based_on_benchmarks(
    benchmark_results: Dict[str, Any],
    current_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    基于基准测试结果优化配置
    
    参数:
        benchmark_results: 基准测试结果
        current_config: 当前配置
        
    返回:
        优化后的配置
    """
    optimized_config = current_config.copy()
    
    # 分析基准测试结果
    if 'data_loading' in benchmark_results:
        dl_result = benchmark_results['data_loading']
        if dl_result.get('avg_batch_load_time_ms', 0) > 100:
            # 数据加载慢，优化配置
            optimized_config['data_loader'] = optimized_config.get('data_loader', {})
            optimized_config['data_loader']['num_workers'] = min(
                optimized_config['data_loader'].get('num_workers', 4) * 2,
                16
            )
            optimized_config['data_loader']['pin_memory'] = True
            optimized_config['data_loader']['prefetch_factor'] = 2
    
    if 'inference' in benchmark_results:
        inf_result = benchmark_results['inference']
        if inf_result.get('avg_inference_time_ms', 0) > 50:
            # 推理慢，优化配置
            optimized_config['inference'] = optimized_config.get('inference', {})
            optimized_config['inference']['use_amp'] = True
            optimized_config['inference']['optimize_model'] = True
    
    if 'memory' in benchmark_results:
        mem_result = benchmark_results['memory']
        if mem_result.get('peak_memory_mb', 0) > 4096:
            # 内存使用高，优化配置
            optimized_config['memory'] = optimized_config.get('memory', {})
            optimized_config['memory']['optimization_level'] = 'high'
            optimized_config['memory']['stream_processing'] = True
            optimized_config['memory']['cache_size'] = max(
                1, optimized_config['memory'].get('cache_size', 100) // 2
            )
    
    return optimized_config


# 导入sys模块
import sys

__all__ = [
    'MetricType',
    'PerformanceMetric',
    'PerformanceMonitor',
    'BenchmarkSuite',
    'performance_context',
    'create_performance_dashboard',
    'optimize_based_on_benchmarks',
]
