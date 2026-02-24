"""
验证集可视化模块

提供ValidationVisualizer类，用于生成对比图像可视化。
"""

import warnings
import os
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
    
    def __init__(self, config: Optional[DiagnosticsConfig] = None, debug: bool = False):
        """
        初始化可视化器
        
        参数:
            config: 诊断配置，如果为None则使用默认配置
            debug: 是否启用调试模式，启用后会打印详细数据信息
        """
        self.config = config or DiagnosticsConfig()
        self.metrics_calculator = ImageMetricsCalculator(config)
        self.debug = debug
    
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
        # 调试模式：打印输入数据信息
        if self.debug:
            self._debug_data(low_dose, enhanced, full_dose, sample_idx)
        
        # 转换为numpy并确保是2D切片
        low_np = self._prepare_image(low_dose)
        enh_np = self._prepare_image(enhanced)
        full_np = self._prepare_image(full_dose)
        
        # 调试模式：打印处理后的数据信息
        if self.debug:
            print(f"处理后的数据形状: low_np={low_np.shape}, enh_np={enh_np.shape}, full_np={full_np.shape}")
            print(f"处理后的数据范围: low_np=[{low_np.min():.6f}, {low_np.max():.6f}], "
                  f"enh_np=[{enh_np.min():.6f}, {enh_np.max():.6f}], "
                  f"full_np=[{full_np.min():.6f}, {full_np.max():.6f}]")
        
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
        
        # 辅助函数：计算显示范围（使用百分比裁剪避免异常值）
        def percentile_clip(data, lower=1, upper=99):
            """使用百分比裁剪计算显示范围"""
            if data.size == 0:
                return data.min(), data.max()
            vmin = np.percentile(data, lower)
            vmax = np.percentile(data, upper)
            return vmin, vmax
        
        # 绘制低剂量输入 - 改进显示参数
        vmin_low, vmax_low = percentile_clip(low_np)
        im1 = axes[0, 0].imshow(low_np, cmap='gray', vmin=vmin_low, vmax=vmax_low)
        axes[0, 0].set_title('Low-Dose Input')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 绘制增强输出
        vmin_enh, vmax_enh = percentile_clip(enh_np)
        im2 = axes[0, 1].imshow(enh_np, cmap='gray', vmin=vmin_enh, vmax=vmax_enh)
        axes[0, 1].set_title('Enhanced Output')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 绘制全剂量目标
        vmin_full, vmax_full = percentile_clip(full_np)
        im3 = axes[1, 0].imshow(full_np, cmap='gray', vmin=vmin_full, vmax=vmax_full)
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
            try:
                # 确保目录存在
                dir_path = os.path.dirname(save_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                
                plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
                print(f"可视化已保存到: {save_path}")
            except Exception as e:
                print(f"保存可视化失败: {e}")
                warnings.warn(f"无法保存可视化到 {save_path}: {e}")
        
        if show:
            plt.show()
        elif not save_path:
            plt.close(fig)
            return None
        
        return fig
    
    def _prepare_image(self, image: torch.Tensor) -> np.ndarray:
        """准备图像用于可视化（修复版本）"""
        # 转换为numpy
        if isinstance(image, torch.Tensor):
            image_np = image.detach().cpu().numpy()
        else:
            image_np = image
        
        # 处理不同维度 - 修复版
        if image_np.ndim == 5:  # (B, C, D, H, W) - 5D数据
            image_np = image_np[0, 0, image_np.shape[2] // 2]  # 取第一个样本，第一个通道，中间深度
        elif image_np.ndim == 4:  # (C, D, H, W) 或 (C, H, W, D) 或 (B, C, H, W)
            # 检测维度顺序
            # 假设深度维度是最小的维度（对于CT图像通常成立）
            dims = image_np.shape
            if dims[0] == 1:  # 可能是 (C, ...) 格式
                # 检查后三个维度中哪个最小
                spatial_dims = dims[1:]
                min_dim_idx = np.argmin(spatial_dims)
                
                if min_dim_idx == 0:  # 深度在第一维 (C, D, H, W)
                    image_np = image_np[0, dims[1] // 2, :, :]
                elif min_dim_idx == 1:  # 深度在第二维 (C, H, D, W)
                    image_np = image_np[0, :, dims[2] // 2, :]
                else:  # 深度在第三维 (C, H, W, D)
                    image_np = image_np[0, :, :, dims[3] // 2]
            else:
                # 可能是 (B, C, H, W) 格式，没有深度维度
                image_np = image_np[0, 0]  # 取第一个样本，第一个通道
        elif image_np.ndim == 3:  # (C, H, W) 或 (H, W, D) 或 (D, H, W)
            # 更智能的判断：检查哪个维度可能是通道维度
            # 通道维度通常较小（1, 3, 4等）且在第一个或最后一个位置
            if image_np.shape[0] in [1, 3, 4]:  # 可能是通道维度在第一维
                image_np = image_np[0] if image_np.shape[0] == 1 else image_np.mean(axis=0)
            elif image_np.shape[2] in [1, 3, 4]:  # 可能是通道维度在最后一维
                image_np = image_np[:, :, 0] if image_np.shape[2] == 1 else image_np.mean(axis=2)
            else:  # 可能是深度维度
                # 取中间切片，假设深度维度是第一维
                image_np = image_np[image_np.shape[0] // 2]
        elif image_np.ndim == 2:  # (H, W)
            pass  # 已经是2D
        else:
            # 对于其他维度，尝试展平或取第一个元素
            warnings.warn(f"无法处理的图像维度: {image_np.ndim}，形状: {image_np.shape}")
            if image_np.ndim > 2:
                # 尝试取第一个元素
                image_np = image_np.reshape(-1, image_np.shape[-2], image_np.shape[-1])[0]
        
        # 改进的归一化
        img_min = image_np.min()
        img_max = image_np.max()
        img_range = img_max - img_min
        
        if img_range > 1e-8:
            image_np = (image_np - img_min) / img_range
        else:
            # 即使范围很小，也进行归一化到 [0, 1]
            # 避免除以零，使用微小值
            image_np = (image_np - img_min) / max(img_range, 1e-8)
        
        return image_np
    
    def _debug_data(self, low_dose: torch.Tensor, enhanced: torch.Tensor,
                   full_dose: torch.Tensor, sample_idx: int = 0):
        """
        调试数据信息
        
        参数:
            low_dose: 低剂量输入图像
            enhanced: 增强输出图像
            full_dose: 全剂量目标图像
            sample_idx: 样本索引
        """
        print("=" * 60)
        print(f"可视化调试信息 - 样本 {sample_idx}")
        print("=" * 60)
        
        for name, tensor in [("low_dose", low_dose), ("enhanced", enhanced), ("full_dose", full_dose)]:
            print(f"{name}:")
            print(f"  形状: {tensor.shape}")
            print(f"  范围: [{tensor.min():.6f}, {tensor.max():.6f}]")
            print(f"  均值: {tensor.mean():.6f}, 标准差: {tensor.std():.6f}")
            if tensor.ndim >= 3:
                if tensor.ndim == 3:
                    print(f"  通道数: {tensor.shape[0]}")
                elif tensor.ndim == 4:
                    print(f"  批次大小: {tensor.shape[0]}, 通道数: {tensor.shape[1]}")
                elif tensor.ndim == 5:
                    print(f"  批次大小: {tensor.shape[0]}, 通道数: {tensor.shape[1]}, 深度: {tensor.shape[2]}")
            
            # 检查是否有NaN或Inf值
            if torch.isnan(tensor).any():
                print(f"  警告: 包含NaN值 ({torch.isnan(tensor).sum()} 个)")
            if torch.isinf(tensor).any():
                print(f"  警告: 包含Inf值 ({torch.isinf(tensor).sum()} 个)")
        
        print("-" * 60)
    
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

