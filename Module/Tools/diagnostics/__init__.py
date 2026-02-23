"""
低剂量CT增强AI诊断工具模块

重构版本：将大文件拆分为多个模块以提高可维护性和性能。

主要功能模块：
1. 配置管理 (config)
2. 图像质量指标计算 (metrics)
3. 验证集可视化 (visualization)
4. 模型诊断 (model)
5. 训练曲线分析 (analysis)
6. 命令行工具 (cli)

使用示例：
    from Module.Tools.diagnostics import DiagnosticsConfig, ImageMetricsCalculator
    config = DiagnosticsConfig()
    calculator = ImageMetricsCalculator(config)
    metrics = calculator.calculate_all_metrics(pred, target)
"""

# 导出主要类和函数以保持向后兼容性
from .config import DiagnosticsConfig
from .metrics import ImageMetricsCalculator

# 导出其他模块（延迟导入以避免循环依赖）
__all__ = [
    'DiagnosticsConfig',
    'ImageMetricsCalculator',
]

# 版本信息
__version__ = '2.0.0'
__author__ = 'AI Team'

# 向后兼容性：导出原始模块中的主要类
def __getattr__(name):
    """延迟导入其他模块以支持按需加载"""
    if name == 'ValidationVisualizer':
        from .visualization import ValidationVisualizer
        return ValidationVisualizer
    elif name == 'ModelDiagnostics':
        from .model import ModelDiagnostics
        return ModelDiagnostics
    elif name == 'TrainingCurveAnalyzer':
        from .analysis import TrainingCurveAnalyzer
        return TrainingCurveAnalyzer
    elif name == 'DiagnosticsCLI':
        from .cli import DiagnosticsCLI
        return DiagnosticsCLI
    else:
        raise AttributeError(f"module 'Module.Tools.diagnostics' has no attribute '{name}'")
