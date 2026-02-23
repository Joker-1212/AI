"""
诊断配置模块

提供DiagnosticsConfig类，用于配置诊断工具的各种参数。
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiagnosticsConfig:
    """诊断配置"""
    # 启用诊断功能
    enable_diagnostics: bool = True
    
    # 指标计算配置
    compute_rmse: bool = True
    compute_mae: bool = True
    compute_psnr: bool = True
    compute_ssim: bool = True
    compute_lpips: bool = True   # 默认开启，需要GPU
    compute_ms_ssim: bool = False  # 默认关闭，因为计算成本高
    
    # 可视化配置
    visualize_samples: int = 5  # 可视化样本数量
    save_visualizations: bool = True
    visualization_dir: str = "./diagnostics/visualizations"
    dpi: int = 150
    visualization_frequency: int = 5  # 每N个epoch可视化一次
    
    # 模型诊断配置
    check_gradients: bool = True
    check_weights: bool = True
    check_activations: bool = False  # 可能影响性能
    check_dead_relu: bool = True
    model_diagnosis_frequency: int = 10  # 每N个epoch诊断一次
    
    # 训练分析配置
    analyze_overfitting: bool = True
    compute_loss_ratio: bool = True
    check_learning_rate: bool = True
    training_analysis_frequency: int = 5  # 每N个epoch分析一次
    
    # 报告配置
    generate_html_report: bool = True
    generate_pdf_report: bool = False
    report_dir: str = "./diagnostics/reports"
    
    def __post_init__(self):
        """确保目录存在"""
        os.makedirs(self.visualization_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
