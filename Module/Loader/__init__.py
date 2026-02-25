"""
数据加载器模块

提供多种数据加载器实现，支持Windows多线程训练优化：
1. 基础数据加载器 (data_loader.py)
2. 优化数据加载器 (optimized_data_loader.py)
3. Windows优化数据加载器 (windows_optimized_loader.py)
"""

from .data_loader import (
    CTDataset,
    get_transforms,
    prepare_data_loaders,
    create_dummy_data,
    test_windows_optimization,
    validate_windows_compatibility
)

from .optimized_data_loader import (
    StreamableCTDataset,
    OptimizedCacheDataset,
    get_optimized_transforms,
    create_optimized_dataloader,
    DataLoaderOptimizer,
    benchmark_dataloader
)

# Windows优化模块（条件导入）
try:
    from .windows_optimized_loader import (
        OptimizationStrategy,
        WindowsOptimizationConfig,
        WindowsCompatibilityChecker,
        PreloadedDataset,
        ThreadPoolDataLoader,
        MemoryMappedDataset,
        WindowsOptimizedDataLoader,
        create_windows_optimized_dataloader,
        benchmark_windows_optimizations
    )
    
    WINDOWS_OPTIMIZATION_AVAILABLE = True
    
except ImportError as e:
    WINDOWS_OPTIMIZATION_AVAILABLE = False
    
    # 创建占位符类
    class OptimizationStrategy:
        NONE = "none"
        PRELOAD = "preload"
        THREADPOOL = "threadpool"
        MEMMAP = "memmap"
        HYBRID = "hybrid"
        AUTO = "auto"
    
    class WindowsOptimizationConfig:
        def __init__(self, **kwargs):
            pass
    
    class WindowsCompatibilityChecker:
        @staticmethod
        def is_windows():
            return False
        
        @staticmethod
        def check_serializability(obj, name="object"):
            return True, None
        
        @staticmethod
        def test_dataset_serializability(dataset, num_samples=3):
            return []
    
    class PreloadedDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("Windows优化模块不可用")
    
    class ThreadPoolDataLoader:
        def __init__(self, *args, **kwargs):
            raise ImportError("Windows优化模块不可用")
    
    class MemoryMappedDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("Windows优化模块不可用")
    
    class WindowsOptimizedDataLoader:
        def __init__(self, *args, **kwargs):
            raise ImportError("Windows优化模块不可用")
    
    def create_windows_optimized_dataloader(*args, **kwargs):
        raise ImportError("Windows优化模块不可用")
    
    def benchmark_windows_optimizations(*args, **kwargs):
        raise ImportError("Windows优化模块不可用")


# 导出所有公共接口
__all__ = [
    # 基础数据加载器
    'CTDataset',
    'get_transforms',
    'prepare_data_loaders',
    'create_dummy_data',
    'test_windows_optimization',
    'validate_windows_compatibility',
    
    # 优化数据加载器
    'StreamableCTDataset',
    'OptimizedCacheDataset',
    'get_optimized_transforms',
    'create_optimized_dataloader',
    'DataLoaderOptimizer',
    'benchmark_dataloader',
    
    # Windows优化数据加载器
    'OptimizationStrategy',
    'WindowsOptimizationConfig',
    'WindowsCompatibilityChecker',
    'PreloadedDataset',
    'ThreadPoolDataLoader',
    'MemoryMappedDataset',
    'WindowsOptimizedDataLoader',
    'create_windows_optimized_dataloader',
    'benchmark_windows_optimizations',
    'WINDOWS_OPTIMIZATION_AVAILABLE',
]


def get_available_loaders():
    """
    获取可用的数据加载器列表
    
    返回:
        数据加载器信息字典
    """
    loaders = {
        'basic': {
            'name': '基础数据加载器',
            'module': 'data_loader',
            'class': 'CTDataset',
            'functions': ['prepare_data_loaders', 'get_transforms'],
            'description': '基础CT数据加载器，支持NIfTI、DICOM、PNG等格式'
        },
        'optimized': {
            'name': '优化数据加载器',
            'module': 'optimized_data_loader',
            'class': 'StreamableCTDataset',
            'functions': ['create_optimized_dataloader', 'benchmark_dataloader'],
            'description': '支持缓存、流式处理和内存优化的数据加载器'
        },
        'windows_optimized': {
            'name': 'Windows优化数据加载器',
            'module': 'windows_optimized_loader',
            'class': 'WindowsOptimizedDataLoader',
            'functions': ['create_windows_optimized_dataloader', 'benchmark_windows_optimizations'],
            'description': '专为Windows系统优化的数据加载器，支持预加载、线程池和内存映射',
            'available': WINDOWS_OPTIMIZATION_AVAILABLE
        }
    }
    
    return loaders


def select_best_loader(dataset_size: int, is_windows: bool = None, memory_limit_mb: float = 1024.0):
    """
    根据条件选择最佳数据加载器
    
    参数:
        dataset_size: 数据集大小（样本数）
        is_windows: 是否为Windows系统（如果为None则自动检测）
        memory_limit_mb: 内存限制（MB）
        
    返回:
        推荐的数据加载器类型和建议
    """
    import platform
    
    if is_windows is None:
        is_windows = platform.system().lower() == "windows"
    
    recommendations = {
        'best_loader': 'basic',
        'reason': '',
        'config': {}
    }
    
    if is_windows:
        if WINDOWS_OPTIMIZATION_AVAILABLE:
            recommendations['best_loader'] = 'windows_optimized'
            
            # 根据数据集大小选择策略
            if dataset_size <= 100:
                strategy = 'preload'
                reason = '小数据集适合完全预加载到内存'
            elif dataset_size <= 1000:
                strategy = 'threadpool'
                reason = '中等数据集适合线程池优化'
            else:
                strategy = 'memmap'
                reason = '大数据集适合内存映射文件'
            
            recommendations['reason'] = f'Windows系统检测到，推荐使用Windows优化加载器（{reason}）'
            recommendations['config'] = {
                'strategy': strategy,
                'max_preload_size_mb': min(memory_limit_mb * 0.8, 4096),
                'threadpool_max_workers': 4
            }
        else:
            recommendations['reason'] = 'Windows系统检测到，但优化模块不可用，使用基础加载器'
            recommendations['config'] = {
                'num_workers': 0,
                'prefetch_factor': None,
                'persistent_workers': False
            }
    else:
        # 非Windows系统
        if dataset_size <= 1000:
            recommendations['best_loader'] = 'optimized'
            recommendations['reason'] = '中小数据集适合使用优化加载器'
            recommendations['config'] = {
                'cache_type': 'memory',
                'num_workers': 4,
                'prefetch_factor': 2
            }
        else:
            recommendations['best_loader'] = 'optimized'
            recommendations['reason'] = '大数据集适合使用优化加载器的流式处理功能'
            recommendations['config'] = {
                'cache_type': 'disk',
                'stream_processing': True,
                'num_workers': 8
            }
    
    return recommendations


# 版本信息
__version__ = '1.2.0'
__author__ = 'CT Reconstruction Team'
__description__ = '低剂量CT数据加载器，支持Windows多线程训练优化'
