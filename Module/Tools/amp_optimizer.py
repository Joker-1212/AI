"""
自动混合精度（AMP）优化器

提供混合精度训练的优化配置和管理工具。
"""

import torch
import warnings
import numpy as np
from typing import Optional, Dict, Any, Tuple, Callable
from contextlib import contextmanager


class AMPConfig:
    """AMP配置类"""
    
    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        cache_enabled: bool = True,
        growth_interval: int = 2000,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        init_scale: float = 65536.0,
        enabled_for_conv: bool = True,
        enabled_for_linear: bool = True,
        enabled_for_matmul: bool = True,
        enabled_for_rnn: bool = False,
    ):
        """
        初始化AMP配置
        
        参数:
            enabled: 是否启用AMP
            dtype: 混合精度数据类型 (float16或bfloat16)
            cache_enabled: 是否启用缓存
            growth_interval: 缩放器增长间隔
            growth_factor: 缩放器增长因子
            backoff_factor: 缩放器回退因子
            init_scale: 初始缩放因子
            enabled_for_conv: 是否为卷积启用
            enabled_for_linear: 是否为线性层启用
            enabled_for_matmul: 是否为矩阵乘法启用
            enabled_for_rnn: 是否为RNN启用
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = dtype
        self.cache_enabled = cache_enabled
        self.growth_interval = growth_interval
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.init_scale = init_scale
        
        # 操作启用配置
        self.enabled_for_conv = enabled_for_conv
        self.enabled_for_linear = enabled_for_linear
        self.enabled_for_matmul = enabled_for_matmul
        self.enabled_for_rnn = enabled_for_rnn
        
        # 验证配置
        self._validate_config()
        
    def _validate_config(self):
        """验证配置"""
        if self.enabled and not torch.cuda.is_available():
            warnings.warn("AMP已启用但CUDA不可用，将禁用AMP")
            self.enabled = False
            
        if self.dtype not in [torch.float16, torch.bfloat16]:
            warnings.warn(f"不支持的AMP数据类型: {self.dtype}，使用float16")
            self.dtype = torch.float16
            
        # 检查bfloat16支持
        if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            warnings.warn("当前设备不支持bfloat16，使用float16")
            self.dtype = torch.float16
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'enabled': self.enabled,
            'dtype': str(self.dtype),
            'cache_enabled': self.cache_enabled,
            'growth_interval': self.growth_interval,
            'growth_factor': self.growth_factor,
            'backoff_factor': self.backoff_factor,
            'init_scale': self.init_scale,
            'enabled_for_conv': self.enabled_for_conv,
            'enabled_for_linear': self.enabled_for_linear,
            'enabled_for_matmul': self.enabled_for_matmul,
            'enabled_for_rnn': self.enabled_for_rnn,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AMPConfig':
        """从字典创建配置"""
        return cls(**config_dict)


class AMPOptimizer:
    """AMP优化器"""
    
    def __init__(self, config: Optional[AMPConfig] = None):
        """
        初始化AMP优化器
        
        参数:
            config: AMP配置
        """
        self.config = config or AMPConfig()
        self.scaler = None
        self._init_scaler()
        
    def _init_scaler(self):
        """初始化梯度缩放器"""
        if self.config.enabled:
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=self.config.enabled,
                init_scale=self.config.init_scale,
                growth_factor=self.config.growth_factor,
                backoff_factor=self.config.backoff_factor,
                growth_interval=self.config.growth_interval,
                enabled_for_cpu=False,
            )
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失"""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer: torch.optim.Optimizer):
        """执行优化器步骤"""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def backward(self, loss: torch.Tensor):
        """反向传播"""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """取消缩放梯度（用于梯度裁剪）"""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
    
    def get_scale(self) -> float:
        """获取当前缩放因子"""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        state = {
            'config': self.config.to_dict(),
        }
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        if 'config' in state_dict:
            self.config = AMPConfig.from_dict(state_dict['config'])
        
        if 'scaler' in state_dict and self.scaler is not None:
            self.scaler.load_state_dict(state_dict['scaler'])
    
    def enable(self):
        """启用AMP"""
        self.config.enabled = True
        self._init_scaler()
    
    def disable(self):
        """禁用AMP"""
        self.config.enabled = False
        self.scaler = None


@contextmanager
def autocast_context(
    enabled: bool = True,
    dtype: Optional[torch.dtype] = None,
    cache_enabled: Optional[bool] = None,
    device_type: str = "cuda"
):
    """
    AMP自动转换上下文管理器
    
    参数:
        enabled: 是否启用
        dtype: 数据类型
        cache_enabled: 是否启用缓存
        device_type: 设备类型
    """
    if not enabled or not torch.cuda.is_available():
        # 如果不启用或CUDA不可用，使用普通上下文
        yield
        return
    
    # 设置autocast参数
    autocast_kwargs = {
        'device_type': device_type,
        'enabled': enabled,
    }
    
    if dtype is not None:
        autocast_kwargs['dtype'] = dtype
    if cache_enabled is not None:
        autocast_kwargs['cache_enabled'] = cache_enabled
    
    # 进入autocast上下文
    with torch.cuda.amp.autocast(**autocast_kwargs):
        yield


def optimize_model_for_amp(
    model: torch.nn.Module,
    config: Optional[AMPConfig] = None
) -> torch.nn.Module:
    """
    为AMP优化模型
    
    参数:
        model: PyTorch模型
        config: AMP配置
        
    返回:
        优化后的模型
    """
    if config is None:
        config = AMPConfig()
    
    if not config.enabled:
        return model
    
    # 设置模型参数为适当的数据类型
    model = model.to(torch.float32)  # 确保模型参数是float32
    
    # 为特定操作启用AMP
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    
    return model


def benchmark_amp_performance(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    num_iterations: int = 100,
    warmup_iterations: int = 10
) -> Dict[str, Any]:
    """
    基准测试AMP性能
    
    参数:
        model: 模型
        input_shape: 输入形状
        num_iterations: 迭代次数
        warmup_iterations: 预热迭代次数
        
    返回:
        性能指标
    """
    import time
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 创建测试输入
    x = torch.randn(input_shape).to(device)
    
    results = {
        'fp32': {'times': [], 'memory': []},
        'amp': {'times': [], 'memory': []},
    }
    
    # 测试FP32性能
    model_fp32 = model.to(torch.float32)
    for i in range(warmup_iterations + num_iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            _ = model_fp32(x)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        if i >= warmup_iterations:
            results['fp32']['times'].append(end_time - start_time)
            results['fp32']['memory'].append(torch.cuda.memory_allocated() / 1024 / 1024)
    
    # 测试AMP性能
    model_amp = model.to(torch.float32)
    amp_optimizer = AMPOptimizer()
    
    for i in range(warmup_iterations + num_iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with autocast_context(enabled=True):
            with torch.no_grad():
                _ = model_amp(x)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        if i >= warmup_iterations:
            results['amp']['times'].append(end_time - start_time)
            results['amp']['memory'].append(torch.cuda.memory_allocated() / 1024 / 1024)
    
    # 计算指标
    metrics = {}
    for mode in ['fp32', 'amp']:
        if results[mode]['times']:
            metrics[f'{mode}_avg_time_ms'] = np.mean(results[mode]['times']) * 1000
            metrics[f'{mode}_std_time_ms'] = np.std(results[mode]['times']) * 1000
            metrics[f'{mode}_avg_memory_mb'] = np.mean(results[mode]['memory'])
            metrics[f'{mode}_min_memory_mb'] = min(results[mode]['memory'])
            metrics[f'{mode}_max_memory_mb'] = max(results[mode]['memory'])
    
    if 'fp32_avg_time_ms' in metrics and 'amp_avg_time_ms' in metrics:
        metrics['speedup'] = metrics['fp32_avg_time_ms'] / metrics['amp_avg_time_ms']
        metrics['memory_saving'] = 1 - (metrics['amp_avg_memory_mb'] / metrics['fp32_avg_memory_mb'])
    
    return metrics


def create_amp_aware_loss_function(
    loss_fn: Callable,
    config: Optional[AMPConfig] = None
) -> Callable:
    """
    创建AMP感知的损失函数
    
    参数:
        loss_fn: 原始损失函数
        config: AMP配置
        
    返回:
        AMP感知的损失函数
    """
    if config is None:
        config = AMPConfig()
    
    def amp_aware_loss(pred, target, *args, **kwargs):
        with autocast_context(enabled=config.enabled):
            return loss_fn(pred, target, *args, **kwargs)
    
    return amp_aware_loss


class AMPTrainingMonitor:
    """AMP训练监控器"""
    
    def __init__(self):
        self.scale_history = []
        self.nan_history = []
        self.inf_history = []
        
    def update(self, scaler: torch.cuda.amp.GradScaler):
        """更新监控数据"""
        if scaler is not None:
            self.scale_history.append(scaler.get_scale())
            
            # 检查NaN/Inf
            if hasattr(scaler, '_found_inf'):
                self.inf_history.append(scaler._found_inf)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.scale_history:
            return {}
        
        return {
            'current_scale': self.scale_history[-1],
            'min_scale': min(self.scale_history),
            'max_scale': max(self.scale_history),
            'avg_scale': np.mean(self.scale_history),
            'scale_changes': len(self.scale_history),
            'nan_count': sum(self.nan_history) if self.nan_history else 0,
            'inf_count': sum(self.inf_history) if self.inf_history else 0,
        }
    
    def should_adjust_config(self, threshold: float = 0.1) -> bool:
        """判断是否应该调整配置"""
        if len(self.scale_history) < 10:
            return False
        
        # 检查缩放因子是否频繁变化
        recent_scales = self.scale_history[-10:]
        scale_variance = np.var(recent_scales) / np.mean(recent_scales)
        
        return scale_variance > threshold


def get_recommended_amp_config(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...]
) -> AMPConfig:
    """
    获取推荐的AMP配置
    
    参数:
        model: 模型
        input_shape: 输入形状
        
    返回:
        推荐的AMP配置
    """
    # 根据模型类型和输入大小推荐配置
    total_params = sum(p.numel() for p in model.parameters())
    input_size = np.prod(input_shape)
    
    # 默认配置
    config = AMPConfig(enabled=True)
    
    # 根据模型大小调整
    if total_params > 100_000_000:  # 大于1亿参数
        # 大模型使用更保守的配置
        config.init_scale = 32768.0
        config.growth_factor = 1.5
        config.backoff_factor = 0.8
    elif total_params < 10_000_000:  # 小于1千万参数
        # 小模型可以使用更激进的配置
        config.init_scale = 131072.0
        config.growth_factor = 2.5
    
    # 根据输入大小调整
    if input_size > 256 * 256 * 32:  # 大输入
        config.cache_enabled = False  # 禁用缓存以节省内存
    
    # 检查bfloat16支持
    if torch.cuda.is_bf16_supported():
        config.dtype = torch.bfloat16
    
    return config


__all__ = [
    'AMPConfig',
    'AMPOptimizer',
    'autocast_context',
    'optimize_model_for_amp',
    'benchmark_amp_performance',
    'create_amp_aware_loss_function',
    'AMPTrainingMonitor',
    'get_recommended_amp_config',
]
