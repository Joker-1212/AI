"""
配置参数 - 支持环境变量和YAML配置
"""
import os
import sys
import platform
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Any, Dict
import yaml

@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = field(default_factory=lambda: os.getenv("CT_DATA_DIR", "./data"))
    low_dose_dir: str = field(default_factory=lambda: os.getenv("CT_LOW_DOSE_DIR", "qd"))
    full_dose_dir: str = field(default_factory=lambda: os.getenv("CT_FULL_DOSE_DIR", "fd"))
    image_size: Tuple[int, int, int] = (512, 512, 1)  # 图像尺寸 (H, W, D)
    batch_size: int = field(default_factory=lambda: int(os.getenv("CT_BATCH_SIZE", "4")))
    num_workers: int = field(default_factory=lambda: int(os.getenv("CT_NUM_WORKERS", "0")))
    pin_memory: bool = field(default_factory=lambda: os.getenv("CT_PIN_MEMORY", "false").lower() == "true")
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    normalize: bool = True
    normalize_range: Tuple[float, float] = (-1000, 1000)  # CT值的HU范围

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = field(default_factory=lambda: os.getenv("CT_MODEL_NAME", "WaveletDomainCNN"))
    in_channels: int = 1
    out_channels: int = 1
    features: Tuple[int, ...] = (32, 64, 128, 256)  # 特征通道数
    dropout: float = 0.1
    use_batch_norm: bool = True
    activation: str = "ReLU"

@dataclass
class TrainingConfig:
    """训练配置"""
    learning_rate: float = field(default_factory=lambda: float(os.getenv("CT_LEARNING_RATE", "1e-4")))
    num_epochs: int = field(default_factory=lambda: int(os.getenv("CT_NUM_EPOCHS", "200")))
    optimizer: str = field(default_factory=lambda: os.getenv("CT_OPTIMIZER", "AdamW"))
    weight_decay: float = field(default_factory=lambda: float(os.getenv("CT_WEIGHT_DECAY", "1e-5")))
    
    # 损失函数配置
    loss_function: str = field(default_factory=lambda: os.getenv("CT_LOSS_FUNCTION", "MixedLoss"))
    loss_weights: Tuple[float, ...] = (1.0, 0.5, 0.1)  # 混合损失权重 [L1, SSIM, Perceptual]
    use_multi_scale_loss: bool = True  # 是否使用多尺度损失
    multi_scale_weights: Tuple[float, ...] = (1.0, 0.5, 0.25)  # 多尺度损失权重
    
    # 学习率调度
    scheduler: str = field(default_factory=lambda: os.getenv("CT_SCHEDULER", "CosineWarmRestarts"))
    patience: int = 15  # ReduceLROnPlateau的耐心值
    min_lr: float = 1e-6
    scheduler_factor: float = 0.5
    scheduler_step_size: int = 20
    scheduler_gamma: float = 0.5
    scheduler_milestones: Tuple[int, ...] = (50, 100, 150)
    
    # 高级训练策略
    warmup_epochs: int = 5  # 学习率热身轮数
    gradient_clip_value: Optional[float] = 1.0  # 梯度裁剪值
    gradient_clip_norm: Optional[float] = None  # 梯度裁剪范数
    use_early_stopping: bool = True  # 是否使用早停
    early_stopping_patience: int = 30  # 早停耐心值
    
    # 混合精度训练
    use_amp: bool = False  # 是否使用自动混合精度
    
    # 日志和监控
    log_interval: int = 10  # 日志记录间隔（批次）
    save_interval: int = 10  # 保存检查点间隔（轮数）
    monitor_metrics: Tuple[str, ...] = ("loss", "psnr", "ssim")  # 监控指标
    
    # 路径配置
    checkpoint_dir: str = field(default_factory=lambda: os.getenv("CT_CHECKPOINT_DIR", "./models/advanced_checkpoints"))
    log_dir: str = field(default_factory=lambda: os.getenv("CT_LOG_DIR", "./logs/advanced_training"))
    device: str = field(default_factory=lambda: os.getenv("CT_DEVICE", "cuda"))

@dataclass
class DiagnosticsConfig:
    """诊断配置"""
    # 启用诊断功能
    enable_diagnostics: bool = field(default_factory=lambda: os.getenv("CT_ENABLE_DIAGNOSTICS", "true").lower() == "true")
    
    # 指标计算配置
    compute_rmse: bool = field(default_factory=lambda: os.getenv("CT_COMPUTE_RMSE", "true").lower() == "true")
    compute_mae: bool = field(default_factory=lambda: os.getenv("CT_COMPUTE_MAE", "true").lower() == "true")
    compute_psnr: bool = field(default_factory=lambda: os.getenv("CT_COMPUTE_PSNR", "true").lower() == "true")
    compute_ssim: bool = field(default_factory=lambda: os.getenv("CT_COMPUTE_SSIM", "true").lower() == "true")
    compute_ms_ssim: bool = field(default_factory=lambda: os.getenv("CT_COMPUTE_MS_SSIM", "false").lower() == "true")
    compute_lpips: bool = field(default_factory=lambda: os.getenv("CT_COMPUTE_LPIPS", "false").lower() == "true")
    
    # 可视化配置
    visualize_samples: int = field(default_factory=lambda: int(os.getenv("CT_VISUALIZE_SAMPLES", "5")))
    save_visualizations: bool = field(default_factory=lambda: os.getenv("CT_SAVE_VISUALIZATIONS", "true").lower() == "true")
    visualization_dir: str = field(default_factory=lambda: os.getenv("CT_VISUALIZATION_DIR", "./diagnostics/visualizations"))
    dpi: int = field(default_factory=lambda: int(os.getenv("CT_DPI", "150")))
    visualization_frequency: int = field(default_factory=lambda: int(os.getenv("CT_VISUALIZATION_FREQUENCY", "5")))
    
    # 模型诊断配置
    check_gradients: bool = field(default_factory=lambda: os.getenv("CT_CHECK_GRADIENTS", "true").lower() == "true")
    check_weights: bool = field(default_factory=lambda: os.getenv("CT_CHECK_WEIGHTS", "true").lower() == "true")
    check_activations: bool = field(default_factory=lambda: os.getenv("CT_CHECK_ACTIVATIONS", "false").lower() == "true")
    check_dead_relu: bool = field(default_factory=lambda: os.getenv("CT_CHECK_DEAD_RELU", "true").lower() == "true")
    model_diagnosis_frequency: int = field(default_factory=lambda: int(os.getenv("CT_MODEL_DIAGNOSIS_FREQUENCY", "10")))
    
    # 训练分析配置
    analyze_overfitting: bool = field(default_factory=lambda: os.getenv("CT_ANALYZE_OVERFITTING", "true").lower() == "true")
    compute_loss_ratio: bool = field(default_factory=lambda: os.getenv("CT_COMPUTE_LOSS_RATIO", "true").lower() == "true")
    check_learning_rate: bool = field(default_factory=lambda: os.getenv("CT_CHECK_LEARNING_RATE", "true").lower() == "true")
    training_analysis_frequency: int = field(default_factory=lambda: int(os.getenv("CT_TRAINING_ANALYSIS_FREQUENCY", "5")))
    
    # 报告配置
    generate_html_report: bool = field(default_factory=lambda: os.getenv("CT_GENERATE_HTML_REPORT", "true").lower() == "true")
    generate_pdf_report: bool = field(default_factory=lambda: os.getenv("CT_GENERATE_PDF_REPORT", "false").lower() == "true")
    report_dir: str = field(default_factory=lambda: os.getenv("CT_REPORT_DIR", "./diagnostics/reports"))
    
    def __post_init__(self):
        """确保目录存在"""
        import os
        os.makedirs(self.visualization_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)


@dataclass
class WindowsOptimizationConfig:
    """Windows多线程优化配置"""
    # Windows系统优化策略
    enabled: bool = field(default_factory=lambda: os.getenv("CT_WINDOWS_OPTIMIZATION", "true").lower() == "true")
    enable_windows_optimization: bool = field(default_factory=lambda: os.getenv("CT_WINDOWS_OPTIMIZATION", "true").lower() == "true")
    platform_detection: str = field(default_factory=lambda: os.getenv("CT_PLATFORM_DETECTION", "auto"))
    
    # 数据加载优化
    data_loader_type: str = field(default_factory=lambda: os.getenv("CT_DATA_LOADER_TYPE", "windows_optimized"))
    use_memory_mapped_files: bool = field(default_factory=lambda: os.getenv("CT_USE_MEMORY_MAPPED_FILES", "true").lower() == "true")
    memory_map_cache_size: int = field(default_factory=lambda: int(os.getenv("CT_MEMORY_MAP_CACHE_SIZE", "1024")))
    preload_data_to_ram: bool = field(default_factory=lambda: os.getenv("CT_PRELOAD_DATA_TO_RAM", "false").lower() == "true")
    preload_batch_count: int = field(default_factory=lambda: int(os.getenv("CT_PRELOAD_BATCH_COUNT", "100")))
    
    # 线程池配置
    num_workers: int = field(default_factory=lambda: int(os.getenv("CT_NUM_WORKERS", "0")))
    use_thread_pool: bool = field(default_factory=lambda: os.getenv("CT_USE_THREAD_POOL", "true").lower() == "true")
    thread_pool_size: int = field(default_factory=lambda: int(os.getenv("CT_THREAD_POOL_SIZE", "4")))
    thread_priority: str = field(default_factory=lambda: os.getenv("CT_THREAD_PRIORITY", "normal"))
    
    # 内存管理
    pin_memory: bool = field(default_factory=lambda: os.getenv("CT_PIN_MEMORY", "false").lower() == "true")
    memory_allocation_strategy: str = field(default_factory=lambda: os.getenv("CT_MEMORY_ALLOCATION_STRATEGY", "balanced"))
    max_memory_usage_mb: int = field(default_factory=lambda: int(os.getenv("CT_MAX_MEMORY_USAGE_MB", "4096")))
    garbage_collection_frequency: int = field(default_factory=lambda: int(os.getenv("CT_GARBAGE_COLLECTION_FREQUENCY", "100")))
    
    # 性能监控
    enable_performance_monitoring: bool = field(default_factory=lambda: os.getenv("CT_ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true")
    monitor_interval_seconds: int = field(default_factory=lambda: int(os.getenv("CT_MONITOR_INTERVAL_SECONDS", "30")))
    log_performance_metrics: bool = field(default_factory=lambda: os.getenv("CT_LOG_PERFORMANCE_METRICS", "true").lower() == "true")
    performance_log_dir: str = field(default_factory=lambda: os.getenv("CT_PERFORMANCE_LOG_DIR", "./logs/windows_performance"))
    
    # 错误处理
    handle_windows_errors: bool = field(default_factory=lambda: os.getenv("CT_HANDLE_WINDOWS_ERRORS", "true").lower() == "true")
    retry_on_failure: bool = field(default_factory=lambda: os.getenv("CT_RETRY_ON_FAILURE", "true").lower() == "true")
    max_retries: int = field(default_factory=lambda: int(os.getenv("CT_MAX_RETRIES", "3")))
    fallback_to_simple_loader: bool = field(default_factory=lambda: os.getenv("CT_FALLBACK_TO_SIMPLE_LOADER", "true").lower() == "true")
    
    # 优化建议
    optimization_level: str = field(default_factory=lambda: os.getenv("CT_OPTIMIZATION_LEVEL", "balanced"))
    windows_10_plus: bool = field(default_factory=lambda: os.getenv("CT_WINDOWS_10_PLUS", "true").lower() == "true")
    disable_antivirus_interference: bool = field(default_factory=lambda: os.getenv("CT_DISABLE_ANTIVIRUS_INTERFERENCE", "true").lower() == "true")
    use_ssd_for_data: bool = field(default_factory=lambda: os.getenv("CT_USE_SSD_FOR_DATA", "true").lower() == "true")
    disable_windows_defender_realtime: bool = field(default_factory=lambda: os.getenv("CT_DISABLE_WINDOWS_DEFENDER_REALTIME", "false").lower() == "true")
    
    # 调试和诊断
    enable_debug_logging: bool = field(default_factory=lambda: os.getenv("CT_ENABLE_DEBUG_LOGGING", "false").lower() == "true")
    log_thread_activity: bool = field(default_factory=lambda: os.getenv("CT_LOG_THREAD_ACTIVITY", "false").lower() == "true")
    profile_data_loading: bool = field(default_factory=lambda: os.getenv("CT_PROFILE_DATA_LOADING", "false").lower() == "true")
    
    @property
    def enabled(self) -> bool:
        """enabled属性的别名，与enable_windows_optimization同步"""
        return self.enable_windows_optimization
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.enable_windows_optimization = value
    
    def __post_init__(self):
        """Windows配置验证"""
        import os
        
        # 同步enabled和enable_windows_optimization字段
        # 如果enabled被显式设置，则用它更新enable_windows_optimization
        # 如果enable_windows_optimization被显式设置，则用它更新enabled
        # 由于dataclass无法知道哪个被显式设置，我们让enabled优先
        # 但为了保持一致性，我们确保两个字段值相同
        self.enable_windows_optimization = self.enabled
        
        # 验证平台检测
        if self.platform_detection not in ["auto", "windows", "linux", "macos"]:
            raise ValueError(f"平台检测必须是auto/windows/linux/macos，当前为{self.platform_detection}")
        
        # 验证数据加载器类型
        if self.data_loader_type not in ["standard", "windows_optimized", "memory_mapped"]:
            raise ValueError(f"数据加载器类型必须是standard/windows_optimized/memory_mapped，当前为{self.data_loader_type}")
        
        # 验证内存映射缓存大小
        if self.memory_map_cache_size < 0:
            raise ValueError(f"内存映射缓存大小必须为非负数，当前为{self.memory_map_cache_size}")
        
        # 验证线程池大小
        if self.thread_pool_size < 0:
            raise ValueError(f"线程池大小必须为非负数，当前为{self.thread_pool_size}")
        
        # 验证线程优先级
        if self.thread_priority not in ["low", "normal", "high", "realtime"]:
            raise ValueError(f"线程优先级必须是low/normal/high/realtime，当前为{self.thread_priority}")
        
        # 验证内存分配策略
        if self.memory_allocation_strategy not in ["conservative", "balanced", "aggressive"]:
            raise ValueError(f"内存分配策略必须是conservative/balanced/aggressive，当前为{self.memory_allocation_strategy}")
        
        # 验证优化级别
        if self.optimization_level not in ["minimal", "balanced", "aggressive"]:
            raise ValueError(f"优化级别必须是minimal/balanced/aggressive，当前为{self.optimization_level}")
        
        # 确保性能日志目录存在
        os.makedirs(self.performance_log_dir, exist_ok=True)
        
        # 自动检测Windows平台
        if self.platform_detection == "auto":
            self._detect_platform()
    
    def _detect_platform(self):
        """自动检测平台"""
        system = platform.system().lower()
        if "windows" in system:
            self.platform_detection = "windows"
        elif "linux" in system:
            self.platform_detection = "linux"
        elif "darwin" in system:
            self.platform_detection = "macos"
        else:
            self.platform_detection = "unknown"
    
    def is_windows(self) -> bool:
        """检查是否为Windows平台"""
        return self.platform_detection == "windows"
    
    def get_recommended_settings(self) -> Dict[str, Any]:
        """获取推荐设置"""
        return {
            "num_workers": 0 if self.is_windows() else 4,
            "pin_memory": not self.is_windows(),
            "data_loader_type": "windows_optimized" if self.is_windows() else "standard",
            "use_memory_mapped_files": self.is_windows(),
            "memory_allocation_strategy": "balanced" if self.is_windows() else "aggressive"
        }
    
    def apply_windows_optimizations(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """应用Windows优化到配置字典"""
        if not self.is_windows() or not self.enable_windows_optimization:
            return config_dict
        
        # 应用Windows特定优化
        optimized = config_dict.copy()
        
        # 数据加载优化
        if "data" not in optimized:
            optimized["data"] = {}
        
        optimized["data"]["num_workers"] = 0
        optimized["data"]["pin_memory"] = False
        
        # 添加Windows优化配置
        if "windows_optimization" not in optimized:
            optimized["windows_optimization"] = {}
        
        windows_config = {
            "enable_windows_optimization": self.enable_windows_optimization,
            "data_loader_type": self.data_loader_type,
            "use_memory_mapped_files": self.use_memory_mapped_files,
            "memory_map_cache_size": self.memory_map_cache_size,
            "num_workers": self.num_workers,
            "use_thread_pool": self.use_thread_pool,
            "thread_pool_size": self.thread_pool_size,
            "enable_performance_monitoring": self.enable_performance_monitoring,
            "handle_windows_errors": self.handle_windows_errors
        }
        
        optimized["windows_optimization"].update(windows_config)
        
        return optimized


@dataclass
class Config:
    """总配置 - 支持从YAML文件和环境变量加载"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    windows_optimization: WindowsOptimizationConfig = field(default_factory=WindowsOptimizationConfig)
    
    def __post_init__(self):
        """配置验证"""
        # 确保分割比例总和为1
        total = self.data.train_split + self.data.val_split + self.data.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"分割比例总和应为1，当前为{total}")
        
        # 确保学习率为浮点数
        try:
            learning_rate = float(self.training.learning_rate)
            self.training.learning_rate = learning_rate
        except (ValueError, TypeError):
            raise ValueError(f"学习率必须是数字，当前为{self.training.learning_rate}")
        
        # 验证学习率范围
        if self.training.learning_rate <= 0:
            raise ValueError(f"学习率必须为正数，当前为{self.training.learning_rate}")
        
        # 确保批次大小为整数
        try:
            batch_size = int(self.data.batch_size)
            self.data.batch_size = batch_size
        except (ValueError, TypeError):
            raise ValueError(f"批次大小必须是整数，当前为{self.data.batch_size}")
        
        # 验证批次大小
        if self.data.batch_size <= 0:
            raise ValueError(f"批次大小必须为正数，当前为{self.data.batch_size}")
        
        # 确保epoch数为整数
        try:
            num_epochs = int(self.training.num_epochs)
            self.training.num_epochs = num_epochs
        except (ValueError, TypeError):
            raise ValueError(f"训练轮数必须是整数，当前为{self.training.num_epochs}")
        
        # 验证epoch数
        if self.training.num_epochs <= 0:
            raise ValueError(f"训练轮数必须为正数，当前为{self.training.num_epochs}")
        
        # 验证设备
        if self.training.device not in ["cuda", "cpu"]:
            raise ValueError(f"设备必须是'cuda'或'cpu'，当前为{self.training.device}")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """从YAML文件加载配置"""
        import os
        from typing import Dict, Any
        
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        # 递归更新配置，支持环境变量覆盖
        def update_with_env_vars(config_dict: Dict[str, Any], prefix: str = "CT_") -> Dict[str, Any]:
            """用环境变量更新配置字典"""
            updated = {}
            for key, value in config_dict.items():
                env_key = f"{prefix}{key.upper()}"
                if isinstance(value, dict):
                    # 递归处理嵌套字典
                    updated[key] = update_with_env_vars(value, f"{env_key}_")
                else:
                    # 检查环境变量
                    env_value = os.getenv(env_key)
                    if env_value is not None:
                        # 尝试转换类型
                        try:
                            if isinstance(value, bool):
                                updated[key] = env_value.lower() == "true"
                            elif isinstance(value, int):
                                updated[key] = int(env_value)
                            elif isinstance(value, float):
                                updated[key] = float(env_value)
                            elif isinstance(value, list):
                                # 尝试解析列表
                                if env_value.startswith("[") and env_value.endswith("]"):
                                    import ast
                                    updated[key] = ast.literal_eval(env_value)
                                else:
                                    updated[key] = [v.strip() for v in env_value.split(",")]
                            else:
                                updated[key] = env_value
                        except (ValueError, SyntaxError):
                            updated[key] = env_value
                    else:
                        updated[key] = value
            return updated
        
        # 应用环境变量覆盖
        yaml_data = update_with_env_vars(yaml_data)
        
        # 创建配置对象
        config = cls()
        
        # 更新数据配置
        if "data" in yaml_data:
            for key, value in yaml_data["data"].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        # 更新模型配置
        if "model" in yaml_data:
            for key, value in yaml_data["model"].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        # 更新训练配置
        if "training" in yaml_data:
            for key, value in yaml_data["training"].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        # 更新诊断配置
        if "diagnostics" in yaml_data:
            for key, value in yaml_data["diagnostics"].items():
                if hasattr(config.diagnostics, key):
                    setattr(config.diagnostics, key, value)
        
        # 更新Windows优化配置
        if "windows_optimization" in yaml_data:
            for key, value in yaml_data["windows_optimization"].items():
                if hasattr(config.windows_optimization, key):
                    setattr(config.windows_optimization, key, value)
        
        # 自动应用Windows优化（如果检测到Windows平台）
        config._apply_windows_optimizations()
        
        # 重新运行验证
        config.__post_init__()
        return config
    
    def _apply_windows_optimizations(self):
        """自动应用Windows优化配置"""
        if not self.windows_optimization.is_windows():
            return
        
        if not self.windows_optimization.enable_windows_optimization:
            return
        
        # 应用Windows特定的数据加载优化
        self.data.num_workers = 0
        self.data.pin_memory = False
        
        # 应用Windows优化建议
        recommended = self.windows_optimization.get_recommended_settings()
        
        # 更新数据配置
        if "num_workers" in recommended:
            self.data.num_workers = recommended["num_workers"]
        if "pin_memory" in recommended:
            self.data.pin_memory = recommended["pin_memory"]
        
        print(f"已应用Windows优化配置: num_workers={self.data.num_workers}, pin_memory={self.data.pin_memory}")
    
    def to_yaml(self, yaml_path: str) -> None:
        """将配置保存到YAML文件"""
        import os
        
        config_dict = {
            "data": {
                "data_dir": self.data.data_dir,
                "low_dose_dir": self.data.low_dose_dir,
                "full_dose_dir": self.data.full_dose_dir,
                "image_size": list(self.data.image_size),
                "batch_size": self.data.batch_size,
                "num_workers": self.data.num_workers,
                "train_split": self.data.train_split,
                "val_split": self.data.val_split,
                "test_split": self.data.test_split,
                "normalize": self.data.normalize,
                "normalize_range": list(self.data.normalize_range),
            },
            "model": {
                "model_name": self.model.model_name,
                "in_channels": self.model.in_channels,
                "out_channels": self.model.out_channels,
                "features": list(self.model.features),
                "dropout": self.model.dropout,
                "use_batch_norm": self.model.use_batch_norm,
                "activation": self.model.activation,
            },
            "training": {
                "learning_rate": self.training.learning_rate,
                "num_epochs": self.training.num_epochs,
                "optimizer": self.training.optimizer,
                "weight_decay": self.training.weight_decay,
                "loss_function": self.training.loss_function,
                "loss_weights": list(self.training.loss_weights),
                "use_multi_scale_loss": self.training.use_multi_scale_loss,
                "multi_scale_weights": list(self.training.multi_scale_weights),
                "scheduler": self.training.scheduler,
                "patience": self.training.patience,
                "min_lr": self.training.min_lr,
                "scheduler_factor": self.training.scheduler_factor,
                "scheduler_step_size": self.training.scheduler_step_size,
                "scheduler_gamma": self.training.scheduler_gamma,
                "scheduler_milestones": list(self.training.scheduler_milestones),
                "warmup_epochs": self.training.warmup_epochs,
                "gradient_clip_value": self.training.gradient_clip_value,
                "gradient_clip_norm": self.training.gradient_clip_norm,
                "use_early_stopping": self.training.use_early_stopping,
                "early_stopping_patience": self.training.early_stopping_patience,
                "use_amp": self.training.use_amp,
                "log_interval": self.training.log_interval,
                "save_interval": self.training.save_interval,
                "monitor_metrics": list(self.training.monitor_metrics),
                "checkpoint_dir": self.training.checkpoint_dir,
                "log_dir": self.training.log_dir,
                "device": self.training.device,
            },
            "diagnostics": {
                "enable_diagnostics": self.diagnostics.enable_diagnostics,
                "compute_rmse": self.diagnostics.compute_rmse,
                "compute_mae": self.diagnostics.compute_mae,
                "compute_psnr": self.diagnostics.compute_psnr,
                "compute_ssim": self.diagnostics.compute_ssim,
                "compute_ms_ssim": self.diagnostics.compute_ms_ssim,
                "compute_lpips": self.diagnostics.compute_lpips,
                "visualize_samples": self.diagnostics.visualize_samples,
                "save_visualizations": self.diagnostics.save_visualizations,
                "visualization_dir": self.diagnostics.visualization_dir,
                "dpi": self.diagnostics.dpi,
                "visualization_frequency": self.diagnostics.visualization_frequency,
                "check_gradients": self.diagnostics.check_gradients,
                "check_weights": self.diagnostics.check_weights,
                "check_activations": self.diagnostics.check_activations,
                "check_dead_relu": self.diagnostics.check_dead_relu,
                "model_diagnosis_frequency": self.diagnostics.model_diagnosis_frequency,
                "analyze_overfitting": self.diagnostics.analyze_overfitting,
                "compute_loss_ratio": self.diagnostics.compute_loss_ratio,
                "check_learning_rate": self.diagnostics.check_learning_rate,
                "training_analysis_frequency": self.diagnostics.training_analysis_frequency,
                "generate_html_report": self.diagnostics.generate_html_report,
                "generate_pdf_report": self.diagnostics.generate_pdf_report,
                "report_dir": self.diagnostics.report_dir,
            },
            "windows_optimization": {
                "enable_windows_optimization": self.windows_optimization.enable_windows_optimization,
                "platform_detection": self.windows_optimization.platform_detection,
                "data_loader_type": self.windows_optimization.data_loader_type,
                "use_memory_mapped_files": self.windows_optimization.use_memory_mapped_files,
                "memory_map_cache_size": self.windows_optimization.memory_map_cache_size,
                "preload_data_to_ram": self.windows_optimization.preload_data_to_ram,
                "preload_batch_count": self.windows_optimization.preload_batch_count,
                "num_workers": self.windows_optimization.num_workers,
                "use_thread_pool": self.windows_optimization.use_thread_pool,
                "thread_pool_size": self.windows_optimization.thread_pool_size,
                "thread_priority": self.windows_optimization.thread_priority,
                "pin_memory": self.windows_optimization.pin_memory,
                "memory_allocation_strategy": self.windows_optimization.memory_allocation_strategy,
                "max_memory_usage_mb": self.windows_optimization.max_memory_usage_mb,
                "garbage_collection_frequency": self.windows_optimization.garbage_collection_frequency,
                "enable_performance_monitoring": self.windows_optimization.enable_performance_monitoring,
                "monitor_interval_seconds": self.windows_optimization.monitor_interval_seconds,
                "log_performance_metrics": self.windows_optimization.log_performance_metrics,
                "performance_log_dir": self.windows_optimization.performance_log_dir,
                "handle_windows_errors": self.windows_optimization.handle_windows_errors,
                "retry_on_failure": self.windows_optimization.retry_on_failure,
                "max_retries": self.windows_optimization.max_retries,
                "fallback_to_simple_loader": self.windows_optimization.fallback_to_simple_loader,
                "optimization_level": self.windows_optimization.optimization_level,
                "windows_10_plus": self.windows_optimization.windows_10_plus,
                "disable_antivirus_interference": self.windows_optimization.disable_antivirus_interference,
                "use_ssd_for_data": self.windows_optimization.use_ssd_for_data,
                "disable_windows_defender_realtime": self.windows_optimization.disable_windows_defender_realtime,
                "enable_debug_logging": self.windows_optimization.enable_debug_logging,
                "log_thread_activity": self.windows_optimization.log_thread_activity,
                "profile_data_loading": self.windows_optimization.profile_data_loading,
            }
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        print(f"配置已保存到: {yaml_path}")
