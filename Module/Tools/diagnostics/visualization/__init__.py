"""
验证集可视化模块

提供ValidationVisualizer类，用于生成对比图像可视化。
"""

import warnings
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..config import DiagnosticsConfig
from ..metrics import ImageMetricsCalculator


class ValidationVisualizer:
    """
    验证集可视化工具
    
    生成对比图像：
    - 低剂量输入
    - 增强输出
    - 全剂量目标
    - 差异图（enhanced - full_dose）
    """
    
    def __init__(self, config: Optional[DiagnosticsConfig] = None):
        """
        初始化可视化器
        
        参数:
            config: 诊断配置，如果为None则使用默认配置
        """
        self.config = config or DiagnosticsConfig()
        self.metrics_calculator = ImageMetricsCalculator(config)
    
    def visualize_sample(
        self,
        low_dose: torch.Tensor,
        enhanced: torch.Tensor,
        full_dose: torch.Tensor,
        sample_idx: int = 0,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> Optional[Figure]:
        """
        可视化单个样本
        
        参数:
            low_dose: 低剂量输入图像 (C, H, W) 或 (C, H, W, D)
            enhanced: 增强输出图像，与low_dose形状相同
            full_dose: 全剂量目标图像，与low_dose形状相同
            sample_idx: 样本索引（用于标题）
            save_path: 保存路径，如果为None则不保存
            show: 是否显示图像
            
        返回:
            matplotlib图形对象，如果show=False且save_path=None则返回None
        """
        # 转换为numpy并确保是2D切片
        low_np = self._prepare_image(low_dose)
        enh_np = self._prepare_image(enhanced)
        full_np = self._prepare_image(full_dose)
        
        # 计算差异
        diff_np = enh_np - full_np
        
        # 计算指标
        metrics = self.metrics_calculator.calculate_all_metrics(
            enhanced.unsqueeze(0) if enhanced.dim() == 3 else enhanced,
            full_dose.unsqueeze(0) if full_dose.dim() == 3 else full_dose
        )
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Sample {sample_idx} - diagnostic visuallization', fontsize=16)
        
        # 绘制低剂量输入
        im1 = axes[0, 0].imshow(low_np, cmap='gray')
        axes[0, 0].set_title('Low-Dose Input')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 绘制增强输出
        im2 = axes[0, 1].imshow(enh_np, cmap='gray')
        axes[0, 1].set_title('Enhanced Output')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 绘制全剂量目标
        im3 = axes[1, 0].imshow(full_np, cmap='gray')
        axes[1, 0].set_title('Full-Dose Target')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 绘制差异图
        im4 = axes[1, 1].imshow(diff_np, cmap='coolwarm', vmin=-np.abs(diff_np).max(), vmax=np.abs(diff_np).max())
        axes[1, 1].set_title('Difference (Enhanced - Full-Dose)')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1])
        
        # 添加指标文本
        metrics_text = f"PSNR: {metrics.get('psnr', 0):.2f} dB\nSSIM: {metrics.get('ssim', 0):.3f}\nRMSE: {metrics.get('rmse', 0):.3f}"
        fig.text(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"可视化已保存到: {save_path}")
        
        if show:
            plt.show()
        elif not save_path:
            plt.close(fig)
            return None
        
        return fig
    
    def _prepare_image(self, image: torch.Tensor) -> np.ndarray:
        """准备图像用于可视化"""
        # 转换为numpy
        if isinstance(image, torch.Tensor):
            image_np = image.detach().cpu().numpy()
        else:
            image_np = image
        
        # 处理不同维度
        if image_np.ndim == 4:  # (B, C, H, W)
            image_np = image_np[0, 0]  # 取第一个样本，第一个通道
        elif image_np.ndim == 3:  # (C, H, W) 或 (H, W, D)
            if image_np.shape[0] <= 3:  # 假设是通道维度
                image_np = image_np[0]  # 取第一个通道
            else:  # 假设是深度维度
                image_np = image_np[:, :, image_np.shape[2] // 2]  # 取中间切片
        elif image_np.ndim == 2:  # (H, W)
            pass  # 已经是2D
        
        # 归一化到[0, 1]用于显示
        img_min = image_np.min()
        img_max = image_np.max()
        if img_max - img_min > 1e-8:
            image_np = (image_np - img_min) / (img_max - img_min)
        
        return image_np
    
    def visualize_batch(
        self,
        low_dose_batch: torch.Tensor,
        enhanced_batch: torch.Tensor,
        full_dose_batch: torch.Tensor,
        max_samples: Optional[int] = None,
        save_dir: Optional[str] = None,
        prefix: str = "sample"
    ) -> List[Figure]:
        """
        可视化批量样本
        
        参数:
            low_dose_batch: 低剂量输入图像批次 (B, C, H, W) 或 (B, C, H, W, D)
            enhanced_batch: 增强输出图像批次，形状相同
            full_dose_batch: 全剂量目标图像批次，形状相同
            max_samples: 最大可视化样本数，如果为None则使用配置中的visualize_samples
            save_dir: 保存目录，如果为None则使用配置中的visualization_dir
            prefix: 文件名前缀
            
        返回:
            图形对象列表
        """
        batch_size = low_dose_batch.shape[0]
        max_samples = max_samples or self.config.visualize_samples
        num_samples = min(batch_size, max_samples)
        
        if save_dir is None:
            save_dir = self.config.visualization_dir
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        figures = []
        
        for i in range(num_samples):
            # 提取单个样本
            low_dose = low_dose_batch[i]
            enhanced = enhanced_batch[i]
            full_dose = full_dose_batch[i]
            
            # 生成保存路径
            if self.config.save_visualizations:
                save_path = os.path.join(save_dir, f"{prefix}_{i:03d}.png")
            else:
                save_path = None
            
            # 可视化单个样本
            fig = self.visualize_sample(
                low_dose, enhanced, full_dose,
                sample_idx=i,
                save_path=save_path,
                show=False
            )
            
            if fig is not None:
                figures.append(fig)
                import matplotlib.pyplot as plt
                plt.close(fig)  # 关闭图形以释放内存
        
        print(f"已可视化 {len(figures)} 个样本")
        return figures


__all__ = ['ValidationVisualizer']
