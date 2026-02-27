import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Module.Inference.inference import CTEnhancer

def percentile_clip(image_np, lower_percentile=1, upper_percentile=99):
    """计算图像的百分位数并返回剪辑范围"""
    if image_np.size == 0:
        return image_np.min(), image_np.max()
    lower = np.percentile(image_np, lower_percentile)
    upper = np.percentile(image_np, upper_percentile)
    return lower, upper

def visualize_samples(sample_id: int, checkpoint_id: int, low_np: np.ndarray, full_np:np.ndarray, enh_np:np.ndarray):
    """可视化输入、增强输出和全剂量图像的对比"""
    diff_np = enh_np - full_np
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Sample {sample_id} - diagnostic visuallization', fontsize=16)

    # 绘制低剂量输入
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

    plt.tight_layout()

    os.makedirs(f"./data/diagnostics/{checkpoint_id}", exist_ok=True)
    plt.savefig(f"./data/diagnostics/{checkpoint_id}/Diagnostic_Visuallization_Sample{sample_id}_Epoch{checkpoint_id}.png", dpi=360, bbox_inches='tight')

    plt.show()

    return fig

def main():
    try:
        sample_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1
        checkpoint_id = int(sys.argv[2]) if len(sys.argv) > 2 else -1
        if (checkpoint_id == -1):
            raise ValueError("Please provide enough arguments: sample_id and checkpoint_id")
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        return
    
    try:
        low_np = np.load(f"./data/qd/{sample_id}.npy")
        full_np = np.load(f"./data/fd/{sample_id}.npy")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    enhancer = CTEnhancer(checkpoint_path=f"./models/checkpoints/checkpoint_{checkpoint_id}.pth")
    input_path = f"./data/qd/{sample_id}.npy"
    output_path = f"./data/rst/{checkpoint_id}/{sample_id}.npy"
    os.makedirs(f"./data/rst/{checkpoint_id}", exist_ok=True)
    enhancer.enhance_file(input_path, output_path)
    
    enh_np = np.load(f"./data/rst/{checkpoint_id}/{sample_id}.npy")
    fig = visualize_samples(sample_id, checkpoint_id, low_np, full_np, enh_np)

if __name__ == "__main__":
    main()
