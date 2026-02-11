"""
推理脚本：使用训练好的模型增强低剂量CT图像
"""
import os
import torch
import numpy as np
from PIL import Image
import nibabel as nib
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional
import glob
import tqdm

from ..Config.config import Config, DataConfig, ModelConfig
from ..Models.models import create_model
from ..Loader.data_loader import get_transforms
from ..Tools.utils import load_checkpoint, visualize_results, ensure_dir


class CTEnhancer:
    """CT增强器"""
    
    def __init__(self, checkpoint_path: str, config: Optional[Config] = None):
        """
        参数:
            checkpoint_path: 训练好的模型检查点路径
            config: 配置对象（如果为None，则从检查点加载）
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if config is None:
            # 尝试从检查点目录加载配置
            config_dir = os.path.dirname(checkpoint_path)
            config_path = os.path.join(config_dir, "config.yaml")
            if os.path.exists(config_path):
                from ..Tools.utils import load_config
                config = load_config(config_path, Config)
            else:
                # 使用默认配置
                config = Config()
        
        self.config = config
        self.model = create_model(config.model).to(self.device)
        
        # 加载检查点
        if os.path.exists(checkpoint_path):
            load_checkpoint(checkpoint_path, self.model)
            print(f"已加载模型检查点: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        self.model.eval()
        
        # 数据变换
        self.transform = get_transforms(config.data, is_train=False)
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 确保是3D（如果是2D，添加深度维度）
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # 应用变换
        data = self.transform({"low": image, "full": image})  # 全剂量仅作为占位符
        return data["low"].unsqueeze(0).to(self.device)  # 添加批次维度
    
    def postprocess(self, tensor: torch.Tensor, original_shape: Tuple) -> np.ndarray:
        """后处理张量到图像"""
        # 移除批次和通道维度
        image = tensor.squeeze().cpu().numpy()
        
        # 如果原始是2D，移除深度维度
        if len(original_shape) == 2:
            if image.shape[-1] == 1:
                image = image[:, :, 0]
        
        return image
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        增强单张图像
        
        参数:
            image: 输入低剂量CT图像 (H, W) 或 (H, W, D)
        
        返回:
            enhanced: 增强后的图像
        """
        original_shape = image.shape
        
        # 预处理
        input_tensor = self.preprocess(image)
        
        # 推理
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        # 后处理
        enhanced = self.postprocess(output_tensor, original_shape)
        
        return enhanced
    
    def enhance_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """批量增强图像"""
        enhanced_images = []
        for img in tqdm.tqdm(images, desc="增强图像"):
            enhanced = self.enhance(img)
            enhanced_images.append(enhanced)
        return enhanced_images
    
    def enhance_file(self, input_path: str, output_path: Optional[str] = None):
        """
        增强文件中的图像
        
        参数:
            input_path: 输入文件路径
            output_path: 输出文件路径（如果为None，则自动生成）
        
        返回:
            output_path: 输出文件路径
        """
        # 加载图像
        image = self._load_image(input_path)
        
        # 增强
        enhanced = self.enhance(image)
        
        # 保存
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_enhanced{ext}"
        
        self._save_image(enhanced, output_path, reference_path=input_path)
        
        print(f"增强图像已保存到: {output_path}")
        return output_path
    
    def enhance_directory(self, input_dir: str, output_dir: str, 
                          pattern: str = "*.nii"):
        """
        增强目录中的所有图像
        
        参数:
            input_dir: 输入目录
            output_dir: 输出目录
            pattern: 文件匹配模式
        """
        ensure_dir(output_dir)
        
        # 查找文件
        file_pattern = os.path.join(input_dir, pattern)
        input_files = sorted(glob.glob(file_pattern))
        
        if not input_files:
            print(f"在 {input_dir} 中未找到匹配 {pattern} 的文件")
            return []
        
        print(f"找到 {len(input_files)} 个文件进行增强")
        
        output_files = []
        for input_file in tqdm.tqdm(input_files, desc="处理文件"):
            filename = os.path.basename(input_file)
            output_file = os.path.join(output_dir, filename)
            
            # 增强并保存
            self.enhance_file(input_file, output_file)
            output_files.append(output_file)
        
        return output_files
    
    def _load_image(self, path: str) -> np.ndarray:
        """加载图像"""
        ext = os.path.splitext(path)[1].lower()
        
        if ext in ['.nii', '.nii.gz']:
            # NIfTI格式
            img = nib.load(path)
            data = img.get_fdata()
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            # 2D图像格式
            img = Image.open(path).convert('L')  # 转为灰度
            data = np.array(img)
        elif ext in ['.dcm', '.dicom']:
            # DICOM格式
            try:
                import pydicom
                ds = pydicom.dcmread(path)
                data = ds.pixel_array
            except ImportError:
                raise ImportError("请安装pydicom以读取DICOM文件")
        elif ext in ['.npy']:
            # numpy数组
            data = np.load(path)
        else:
            raise ValueError(f"不支持的图像格式: {ext}")
        
        return data
    
    def _save_image(self, image: np.ndarray, path: str, reference_path: Optional[str] = None):
        """保存图像"""
        ext = os.path.splitext(path)[1].lower()
        
        if ext in ['.nii', '.nii.gz']:
            # NIfTI格式
            if reference_path and os.path.exists(reference_path):
                ref_img = nib.load(reference_path)
                affine = ref_img.affine
                header = ref_img.header
            else:
                affine = np.eye(4)
                header = None
            
            # 确保是3D
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            
            img = nib.Nifti1Image(image, affine, header)
            nib.save(img, path)
        
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            # 2D图像格式
            if len(image.shape) == 3:
                # 取中间切片
                image = image[:, :, image.shape[2]//2]
            
            # 归一化到0-255
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            image = (image * 255).astype(np.uint8)
            
            img = Image.fromarray(image)
            img.save(path)
        
        elif ext in ['.npy']:
            # numpy数组
            np.save(path, image)
        
        else:
            raise ValueError(f"不支持的输出格式: {ext}")


def compare_results(input_dir: str, output_dir: str, num_samples: int = 5):
    """
    比较输入和输出结果
    
    参数:
        input_dir: 输入目录（低剂量图像）
        output_dir: 输出目录（增强图像）
        num_samples: 要可视化的样本数量
    """
    # 查找文件
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.nii")))
    output_files = sorted(glob.glob(os.path.join(output_dir, "*.nii")))
    
    if not input_files or not output_files:
        print("未找到足够的文件进行比较")
        return
    
    # 限制样本数量
    num_samples = min(num_samples, len(input_files), len(output_files))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))
    
    for i in range(num_samples):
        # 加载图像
        input_img = nib.load(input_files[i]).get_fdata()
        output_img = nib.load(output_files[i]).get_fdata()
        
        # 取中间切片
        if len(input_img.shape) == 3:
            slice_idx = input_img.shape[2] // 2
            input_slice = input_img[:, :, slice_idx]
            output_slice = output_img[:, :, slice_idx]
        else:
            input_slice = input_img
            output_slice = output_img
        
        # 绘制
        if num_samples == 1:
            ax1, ax2 = axes
        else:
            ax1, ax2 = axes[i]
        
        im1 = ax1.imshow(input_slice, cmap='gray')
        ax1.set_title(f'低剂量输入 {i+1}')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        im2 = ax2.imshow(output_slice, cmap='gray')
        ax2.set_title(f'增强输出 {i+1}')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数：示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description='低剂量CT图像增强')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图像路径或目录')
    parser.add_argument('--output', type=str, default=None,
                       help='输出路径或目录')
    parser.add_argument('--pattern', type=str, default="*.nii",
                       help='文件匹配模式（当输入为目录时）')
    
    args = parser.parse_args()
    
    # 创建增强器
    enhancer = CTEnhancer(args.checkpoint)
    
    # 检查输入是文件还是目录
    if os.path.isfile(args.input):
        # 单个文件
        output_path = enhancer.enhance_file(args.input, args.output)
        print(f"增强完成: {output_path}")
    
    elif os.path.isdir(args.input):
        # 目录
        if args.output is None:
            args.output = os.path.join(args.input, "enhanced")
        
        output_files = enhancer.enhance_directory(
            args.input, args.output, args.pattern
        )
        print(f"增强完成，共处理 {len(output_files)} 个文件")
        
        # 生成比较图
        compare_results(args.input, args.output)
    
    else:
        print(f"输入路径不存在: {args.input}")


if __name__ == "__main__":
    main()
