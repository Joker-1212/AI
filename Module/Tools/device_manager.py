"""
设备管理器

提供智能设备选择、GPU内存管理和设备优化功能。
"""

import torch
import warnings
from typing import Optional, Dict, Any, List, Tuple
import gc
import psutil
import platform


class DeviceManager:
    """智能设备管理器"""
    
    def __init__(self, preferred_device: str = "auto", memory_limit_mb: Optional[float] = None):
        """
        初始化设备管理器
        
        参数:
            preferred_device: 首选设备 ('auto', 'cuda', 'cpu', 'mps')
            memory_limit_mb: GPU内存限制（MB）
        """
        self.preferred_device = preferred_device
        self.memory_limit_mb = memory_limit_mb
        self.device_info = self._collect_device_info()
        self.selected_device = self._select_device()
        
    def _collect_device_info(self) -> Dict[str, Any]:
        """收集设备信息"""
        info = {
            'cpu': self._get_cpu_info(),
            'cuda': self._get_cuda_info(),
            'mps': self._get_mps_info(),
            'system': self._get_system_info(),
        }
        return info
        
    def _get_cpu_info(self) -> Dict[str, Any]:
        """获取CPU信息"""
        cpu_info = {
            'available': True,
            'cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'memory_total_mb': psutil.virtual_memory().total / 1024 / 1024,
            'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
        }
        return cpu_info
        
    def _get_cuda_info(self) -> Dict[str, Any]:
        """获取CUDA信息"""
        cuda_info = {
            'available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'devices': [],
        }
        
        if cuda_info['available']:
            for i in range(cuda_info['device_count']):
                device_props = {
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'capability': torch.cuda.get_device_capability(i),
                    'total_memory_mb': torch.cuda.get_device_properties(i).total_memory / 1024 / 1024,
                    'free_memory_mb': torch.cuda.mem_get_info(i)[0] / 1024 / 1024 if torch.cuda.is_available() else 0,
                }
                cuda_info['devices'].append(device_props)
                
        return cuda_info
        
    def _get_mps_info(self) -> Dict[str, Any]:
        """获取MPS（Apple Silicon）信息"""
        mps_info = {
            'available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        }
        
        if mps_info['available']:
            mps_info['built'] = torch.backends.mps.is_built()
            
        return mps_info
        
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
        }
        
    def _select_device(self) -> torch.device:
        """智能选择设备"""
        if self.preferred_device == "auto":
            return self._auto_select_device()
        elif self.preferred_device == "cuda":
            if self.device_info['cuda']['available']:
                return self._select_best_cuda_device()
            else:
                warnings.warn("CUDA不可用，回退到CPU")
                return torch.device("cpu")
        elif self.preferred_device == "mps":
            if self.device_info['mps']['available']:
                return torch.device("mps")
            else:
                warnings.warn("MPS不可用，回退到CPU")
                return torch.device("cpu")
        else:
            return torch.device("cpu")
            
    def _auto_select_device(self) -> torch.device:
        """自动选择最佳设备"""
        # 检查CUDA
        if self.device_info['cuda']['available']:
            # 检查是否有足够的内存
            best_device = self._select_best_cuda_device()
            if best_device.type == 'cuda':
                return best_device
                
        # 检查MPS
        if self.device_info['mps']['available']:
            return torch.device("mps")
            
        # 回退到CPU
        return torch.device("cpu")
        
    def _select_best_cuda_device(self) -> torch.device:
        """选择最佳CUDA设备"""
        if not self.device_info['cuda']['available']:
            return torch.device("cpu")
            
        devices = self.device_info['cuda']['devices']
        if not devices:
            return torch.device("cpu")
            
        # 如果有内存限制，选择满足限制的设备
        if self.memory_limit_mb is not None:
            suitable_devices = [
                d for d in devices 
                if d['free_memory_mb'] >= self.memory_limit_mb
            ]
            if suitable_devices:
                # 选择空闲内存最多的设备
                best = max(suitable_devices, key=lambda d: d['free_memory_mb'])
                return torch.device(f"cuda:{best['index']}")
                
        # 否则选择计算能力最强的设备
        best_device = max(devices, key=lambda d: d['capability'])
        return torch.device(f"cuda:{best_device['index']}")
        
    def get_device(self) -> torch.device:
        """获取选择的设备"""
        return self.selected_device
        
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        return self.device_info
        
    def print_device_info(self):
        """打印设备信息"""
        info = self.device_info
        
        print("=" * 50)
        print("设备信息报告")
        print("=" * 50)
        
        # 系统信息
        print(f"\n系统信息:")
        print(f"  平台: {info['system']['platform']} {info['system']['platform_version']}")
        print(f"  Python: {info['system']['python_version']}")
        print(f"  PyTorch: {info['system']['torch_version']}")
        
        # CPU信息
        cpu = info['cpu']
        print(f"\nCPU信息:")
        print(f"  物理核心: {cpu['cores']}")
        print(f"  逻辑核心: {cpu['logical_cores']}")
        print(f"  内存总量: {cpu['memory_total_mb']:.0f} MB")
        print(f"  可用内存: {cpu['memory_available_mb']:.0f} MB")
        
        # CUDA信息
        cuda = info['cuda']
        print(f"\nCUDA信息:")
        print(f"  可用: {cuda['available']}")
        if cuda['available']:
            print(f"  设备数量: {cuda['device_count']}")
            for i, device in enumerate(cuda['devices']):
                print(f"  设备 {i}: {device['name']}")
                print(f"    计算能力: {device['capability'][0]}.{device['capability'][1]}")
                print(f"    显存总量: {device['total_memory_mb']:.0f} MB")
                print(f"    可用显存: {device['free_memory_mb']:.0f} MB")
                
        # MPS信息
        mps = info['mps']
        print(f"\nMPS信息:")
        print(f"  可用: {mps['available']}")
        
        # 选择的设备
        print(f"\n选择的设备: {self.selected_device}")
        print("=" * 50)
        
    def optimize_memory(self, model: torch.nn.Module, input_shape: Tuple[int, ...]):
        """
        优化模型内存使用
        
        参数:
            model: PyTorch模型
            input_shape: 输入形状
        """
        device = self.selected_device
        
        if device.type == 'cuda':
            self._optimize_cuda_memory(model, input_shape)
        elif device.type == 'mps':
            self._optimize_mps_memory(model, input_shape)
        else:
            self._optimize_cpu_memory(model, input_shape)
            
    def _optimize_cuda_memory(self, model: torch.nn.Module, input_shape: Tuple[int, ...]):
        """优化CUDA内存"""
        # 设置内存限制
        if self.memory_limit_mb is not None:
            torch.cuda.set_per_process_memory_fraction(
                self.memory_limit_mb / self.device_info['cuda']['devices'][0]['total_memory_mb']
            )
            
        # 清理缓存
        torch.cuda.empty_cache()
        
        # 启用cudnn基准测试（如果输入大小固定）
        if all(s > 0 for s in input_shape):
            torch.backends.cudnn.benchmark = True
            
    def _optimize_mps_memory(self, model: torch.nn.Module, input_shape: Tuple[int, ...]):
        """优化MPS内存"""
        # MPS特定的优化
        pass
        
    def _optimize_cpu_memory(self, model: torch.nn.Module, input_shape: Tuple[int, ...]):
        """优化CPU内存"""
        # 设置线程数
        torch.set_num_threads(min(4, self.device_info['cpu']['logical_cores']))
        
    def clear_memory(self):
        """清理内存"""
        gc.collect()
        
        if self.selected_device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elif self.selected_device.type == 'mps':
            if hasattr(torch, 'mps'):
                torch.mps.empty_cache()
                
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计"""
        stats = {
            'cpu': self._get_cpu_memory_stats(),
        }
        
        if self.selected_device.type == 'cuda':
            stats['gpu'] = self._get_gpu_memory_stats()
        elif self.selected_device.type == 'mps':
            stats['mps'] = self._get_mps_memory_stats()
            
        return stats
        
    def _get_cpu_memory_stats(self) -> Dict[str, float]:
        """获取CPU内存统计"""
        memory = psutil.virtual_memory()
        return {
            'total_mb': memory.total / 1024 / 1024,
            'available_mb': memory.available / 1024 / 1024,
            'used_mb': memory.used / 1024 / 1024,
            'percent': memory.percent,
        }
        
    def _get_gpu_memory_stats(self) -> Dict[str, float]:
        """获取GPU内存统计"""
        if not torch.cuda.is_available():
            return {}
            
        torch.cuda.synchronize()
        device_index = self.selected_device.index if self.selected_device.index else 0
        
        return {
            'allocated_mb': torch.cuda.memory_allocated(device_index) / 1024 / 1024,
            'reserved_mb': torch.cuda.memory_reserved(device_index) / 1024 / 1024,
            'max_allocated_mb': torch.cuda.max_memory_allocated(device_index) / 1024 / 1024,
            'free_mb': torch.cuda.mem_get_info(device_index)[0] / 1024 / 1024,
            'total_mb': torch.cuda.mem_get_info(device_index)[1] / 1024 / 1024,
        }
        
    def _get_mps_memory_stats(self) -> Dict[str, float]:
        """获取MPS内存统计"""
        # MPS目前没有公开的内存统计API
        return {}


def get_optimal_device(
    memory_requirement_mb: Optional[float] = None,
    prefer_gpu: bool = True
) -> torch.device:
    """
    获取最优设备（简化接口）
    
    参数:
        memory_requirement_mb: 内存需求（MB）
        prefer_gpu: 是否优先使用GPU
        
    返回:
        最优设备
    """
    manager = DeviceManager(
        preferred_device="auto",
        memory_limit_mb=memory_requirement_mb
    )
    
    if not prefer_gpu:
        manager.preferred_device = "cpu"
        
    return manager.get_device()


def optimize_model_for_device(
    model: torch.nn.Module,
    device: torch.device,
    input_shape: Optional[Tuple[int, ...]] = None
) -> torch.nn.Module:
    """
    为设备优化模型
    
    参数:
        model: PyTorch模型
        device: 目标设备
        input_shape: 输入形状（用于优化）
        
    返回:
        优化后的模型
    """
    # 移动到设备
    model = model.to(device)
    
    # 根据设备类型进行优化
    if device.type == 'cuda':
        # 启用cudnn基准测试
        if input_shape and all(s > 0 for s in input_shape):
            torch.backends.cudnn.benchmark = True
            
        # 使用混合精度（如果可用）
        try:
            from torch.cuda.amp import autocast
            # 模型已经准备好混合精度
        except ImportError:
            pass
            
    elif device.type == 'mps':
        # MPS特定的优化
        pass
        
    else:
        # CPU优化：设置线程数
        torch.set_num_threads(min(4, torch.get_num_threads()))
        
    return model


def monitor_memory_usage(func):
    """
    内存使用监控装饰器
    
    用法:
        @monitor_memory_usage
        def my_function():
            # 内存密集型操作
            pass
    """
    def wrapper(*args, **kwargs):
        from .memory_optimizer import MemoryMonitor
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        monitor = MemoryMonitor(device)
        monitor.start()
        
        try:
            result = func(*args, **kwargs)
            monitor.snapshot("end")
            
            if monitor.detect_leaks():
                warnings.warn(f"函数 {func.__name__} 可能发生内存泄漏")
                
            return result
        finally:
            # 清理内存
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                
    return wrapper


__all__ = [
    'DeviceManager',
    'get_optimal_device',
    'optimize_model_for_device',
    'monitor_memory_usage',
]
