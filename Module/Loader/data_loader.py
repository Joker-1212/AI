"""
低剂量CT数据加载和预处理
"""
import os
import numpy as np
from typing import Tuple, List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensityRange,
    RandRotate,
    RandFlip,
    RandZoom,
    RandGaussianNoise,
    RandAdjustContrast,
    Resize,
    ToTensor,
)
from monai.data import CacheDataset, DataLoader as MonaiDataLoader
import nibabel as nib
from PIL import Image
import glob

from ..Config.config import DataConfig


class CTDataset(Dataset):
    """低剂量CT数据集"""
    
    def __init__(self, low_dose_paths: List[str], full_dose_paths: List[str], 
                 config: DataConfig, transform=None, is_train=True):
        """
        参数:
            low_dose_paths: 低剂量CT图像路径列表
            full_dose_paths: 全剂量CT图像路径列表
            config: 数据配置
            transform: 可选的变换
            is_train: 是否为训练集（决定是否使用数据增强）
        """
        self.low_dose_paths = low_dose_paths
        self.full_dose_paths = full_dose_paths
        self.config = config
        self.transform = transform
        self.is_train = is_train
        
        # 确保路径匹配
        assert len(low_dose_paths) == len(full_dose_paths), \
            "低剂量和全剂量图像数量不匹配"
    
    def __len__(self):
        return len(self.low_dose_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        low_dose = self._load_image(self.low_dose_paths[idx])
        full_dose = self._load_image(self.full_dose_paths[idx])
        
        # 应用变换
        if self.transform:
            data = self.transform({"low": low_dose, "full": full_dose})
            low_dose = data["low"]
            full_dose = data["full"]
        
        return low_dose, full_dose
    
    def _load_image(self, path: str) -> np.ndarray:
        """加载图像，支持NIfTI、DICOM、PNG等格式"""
        ext = os.path.splitext(path)[1].lower()
        
        if ext in ['.nii', '.nii.gz']:
            # NIfTI格式
            img = nib.load(path)
            data = img.get_fdata()
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            # 2D图像格式
            img = Image.open(path).convert('L')  # 转为灰度
            data = np.array(img)
        elif ext in ['.dcm', '.dicom']:
            # DICOM格式（需要pydicom）
            try:
                import pydicom
                ds = pydicom.dcmread(path)
                data = ds.pixel_array
            except ImportError:
                raise ImportError("请安装pydicom以读取DICOM文件")
        else:
            # 尝试作为numpy数组加载
            try:
                data = np.load(path)
            except:
                raise ValueError(f"不支持的图像格式: {ext}")
        
        # 确保是3D（如果是2D，添加深度维度）
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)
        
        return data


def get_transforms(config: DataConfig, is_train: bool = True):
    """
    获取数据变换管道
    """
    transforms = []
    
    # 1. 确保通道优先（MONAI要求）
    transforms.append(EnsureChannelFirst())
    
    # 2. 调整强度范围（CT值标准化）
    if config.normalize:
        transforms.append(
            ScaleIntensityRange(
                a_min=config.normalize_range[0],
                a_max=config.normalize_range[1],
                b_min=0.0,
                b_max=1.0,
                clip=True
            )
        )
    
    # 3. 调整大小
    transforms.append(
        Resize(spatial_size=config.image_size[:2])  # 仅调整H,W，保持深度
    )
    
    # 4. 训练时的数据增强
    if is_train:
        transforms.extend([
            RandRotate(range_x=15, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandFlip(spatial_axis=1, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            RandGaussianNoise(prob=0.2, std=0.01),
            RandAdjustContrast(prob=0.3, gamma=(0.7, 1.3)),
        ])
    
    # 5. 转换为张量
    transforms.append(ToTensor(dtype=torch.float32))
    
    return Compose(transforms)


def prepare_data_loaders(config: DataConfig):
    """
    准备训练、验证和测试数据加载器
    """
    # 查找图像文件
    low_dose_pattern = os.path.join(config.data_dir, config.low_dose_dir, "*")
    full_dose_pattern = os.path.join(config.data_dir, config.full_dose_dir, "*")
    
    low_dose_files = sorted(glob.glob(low_dose_pattern))
    full_dose_files = sorted(glob.glob(full_dose_pattern))
    
    if not low_dose_files or not full_dose_files:
        raise FileNotFoundError(
            f"在 {config.data_dir} 中未找到图像文件。"
            f"请确保 {config.low_dose_dir} 和 {config.full_dose_dir} 目录存在并包含图像。"
        )
    
    # 分割数据集
    n_total = len(low_dose_files)
    n_train = int(n_total * config.train_split)
    n_val = int(n_total * config.val_split)
    n_test = n_total - n_train - n_val
    
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # 创建数据集
    train_dataset = CTDataset(
        [low_dose_files[i] for i in train_idx],
        [full_dose_files[i] for i in train_idx],
        config,
        transform=get_transforms(config, is_train=True),
        is_train=True
    )
    
    val_dataset = CTDataset(
        [low_dose_files[i] for i in val_idx],
        [full_dose_files[i] for i in val_idx],
        config,
        transform=get_transforms(config, is_train=False),
        is_train=False
    )
    
    test_dataset = CTDataset(
        [low_dose_files[i] for i in test_idx],
        [full_dose_files[i] for i in test_idx],
        config,
        transform=get_transforms(config, is_train=False),
        is_train=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_dummy_data(config: DataConfig, num_samples: int = 10):
    """
    创建虚拟数据用于测试
    """
    os.makedirs(os.path.join(config.data_dir, config.low_dose_dir), exist_ok=True)
    os.makedirs(os.path.join(config.data_dir, config.full_dose_dir), exist_ok=True)
    
    for i in range(num_samples):
        # 创建随机CT图像（模拟低剂量和高剂量）
        h, w = config.image_size[:2]
        
        # 低剂量：添加更多噪声
        low_dose = np.random.randn(h, w) * 0.3 + 0.5
        low_dose = np.clip(low_dose, 0, 1)
        
        # 全剂量：更清晰的图像
        full_dose = np.random.randn(h, w) * 0.1 + 0.5
        full_dose = np.clip(full_dose, 0, 1)
        
        # 保存为PNG
        low_img = Image.fromarray((low_dose * 255).astype(np.uint8))
        full_img = Image.fromarray((full_dose * 255).astype(np.uint8))
        
        low_path = os.path.join(config.data_dir, config.low_dose_dir, f"low_{i:03d}.png")
        full_path = os.path.join(config.data_dir, config.full_dose_dir, f"full_{i:03d}.png")
        
        low_img.save(low_path)
        full_img.save(full_path)
    
    print(f"已创建 {num_samples} 个虚拟数据样本到 {config.data_dir}")
