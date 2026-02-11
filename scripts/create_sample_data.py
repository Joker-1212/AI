"""
创建示例数据用于测试
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from PIL import Image
import nibabel as nib
from Module.Config.config import DataConfig
from Module.Loader.data_loader import create_dummy_data


def create_2d_ct_samples(config: DataConfig, num_samples: int = 20):
    """创建2D CT样本"""
    low_dose_dir = os.path.join(config.data_dir, config.low_dose_dir)
    full_dose_dir = os.path.join(config.data_dir, config.full_dose_dir)
    
    os.makedirs(low_dose_dir, exist_ok=True)
    os.makedirs(full_dose_dir, exist_ok=True)
    
    print(f"创建 {num_samples} 个2D CT样本...")
    
    for i in range(num_samples):
        # 创建模拟的CT图像（模拟人体横截面）
        h, w = config.image_size[:2]
        
        # 创建模拟的解剖结构（椭圆表示器官）
        y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
        
        # 模拟身体轮廓（椭圆）
        body_mask = (x**2/(w//2)**2 + y**2/(h//2)**2) <= 1
        
        # 模拟肺部（两个椭圆）
        lung1_mask = ((x + w//4)**2/(w//6)**2 + (y)**2/(h//4)**2) <= 1
        lung2_mask = ((x - w//4)**2/(w//6)**2 + (y)**2/(h//4)**2) <= 1
        
        # 模拟脊柱（圆形）
        spine_mask = (x**2/(w//20)**2 + (y + h//8)**2/(h//10)**2) <= 1
        
        # 创建全剂量CT（清晰的图像）
        full_dose = np.zeros((h, w))
        full_dose[body_mask] = 0.3  # 软组织
        full_dose[lung1_mask | lung2_mask] = 0.1  # 肺部（空气）
        full_dose[spine_mask] = 0.8  # 骨骼
        
        # 添加一些纹理
        texture = np.random.randn(h, w) * 0.02
        full_dose += texture
        full_dose = np.clip(full_dose, 0, 1)
        
        # 创建低剂量CT（添加噪声和模糊）
        low_dose = full_dose.copy()
        # 添加高斯噪声
        low_dose += np.random.randn(h, w) * 0.1
        # 添加模糊（简单的高斯模糊）
        from scipy.ndimage import gaussian_filter
        low_dose = gaussian_filter(low_dose, sigma=1.0)
        low_dose = np.clip(low_dose, 0, 1)
        
        # 保存为PNG
        low_img = Image.fromarray((low_dose * 255).astype(np.uint8))
        full_img = Image.fromarray((full_dose * 255).astype(np.uint8))
        
        low_path = os.path.join(low_dose_dir, f"low_2d_{i:03d}.png")
        full_path = os.path.join(full_dose_dir, f"full_2d_{i:03d}.png")
        
        low_img.save(low_path)
        full_img.save(full_path)
        
        # 同时保存为NIfTI格式
        low_nii = nib.Nifti1Image(low_dose.astype(np.float32), np.eye(4))
        full_nii = nib.Nifti1Image(full_dose.astype(np.float32), np.eye(4))
        
        low_nii_path = os.path.join(low_dose_dir, f"low_2d_{i:03d}.nii")
        full_nii_path = os.path.join(full_dose_dir, f"full_2d_{i:03d}.nii")
        
        nib.save(low_nii, low_nii_path)
        nib.save(full_nii, full_nii_path)
    
    print(f"2D样本已保存到 {config.data_dir}")


def create_3d_ct_samples(config: DataConfig, num_volumes: int = 5, slices_per_volume: int = 32):
    """创建3D CT体积样本"""
    low_dose_dir = os.path.join(config.data_dir, config.low_dose_dir + "_3d")
    full_dose_dir = os.path.join(config.data_dir, config.full_dose_dir + "_3d")
    
    os.makedirs(low_dose_dir, exist_ok=True)
    os.makedirs(full_dose_dir, exist_ok=True)
    
    print(f"创建 {num_volumes} 个3D CT体积，每个 {slices_per_volume} 层...")
    
    for v in range(num_volumes):
        volume_low = []
        volume_full = []
        
        for s in range(slices_per_volume):
            h, w = config.image_size[:2]
            
            # 创建模拟的CT切片，随着深度变化
            y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
            
            # 身体轮廓（随着深度变化的椭圆）
            body_a = w//2 * (1 - 0.2 * np.sin(np.pi * s / slices_per_volume))
            body_b = h//2 * (1 - 0.2 * np.cos(np.pi * s / slices_per_volume))
            body_mask = (x**2/body_a**2 + y**2/body_b**2) <= 1
            
            # 模拟器官（位置随深度变化）
            organ_x = int(w//4 * np.sin(2*np.pi * s / slices_per_volume))
            organ_y = int(h//4 * np.cos(2*np.pi * s / slices_per_volume))
            organ_mask = ((x - organ_x)**2/(w//8)**2 + (y - organ_y)**2/(h//8)**2) <= 1
            
            # 创建全剂量CT
            full_slice = np.zeros((h, w))
            full_slice[body_mask] = 0.3
            full_slice[organ_mask] = 0.6
            full_slice += np.random.randn(h, w) * 0.02
            full_slice = np.clip(full_slice, 0, 1)
            
            # 创建低剂量CT
            low_slice = full_slice.copy()
            low_slice += np.random.randn(h, w) * 0.08
            from scipy.ndimage import gaussian_filter
            low_slice = gaussian_filter(low_slice, sigma=1.2)
            low_slice = np.clip(low_slice, 0, 1)
            
            volume_low.append(low_slice)
            volume_full.append(full_slice)
        
        # 堆叠切片创建3D体积
        volume_low = np.stack(volume_low, axis=-1)  # (H, W, D)
        volume_full = np.stack(volume_full, axis=-1)
        
        # 保存为NIfTI
        low_nii = nib.Nifti1Image(volume_low.astype(np.float32), np.eye(4))
        full_nii = nib.Nifti1Image(volume_full.astype(np.float32), np.eye(4))
        
        low_path = os.path.join(low_dose_dir, f"low_3d_{v:03d}.nii.gz")
        full_path = os.path.join(full_dose_dir, f"full_3d_{v:03d}.nii.gz")
        
        nib.save(low_nii, low_path)
        nib.save(full_nii, full_path)
        
        # 保存中间切片为PNG用于可视化
        mid_slice = slices_per_volume // 2
        low_slice_img = Image.fromarray((volume_low[:, :, mid_slice] * 255).astype(np.uint8))
        full_slice_img = Image.fromarray((volume_full[:, :, mid_slice] * 255).astype(np.uint8))
        
        low_slice_path = os.path.join(low_dose_dir, f"low_3d_{v:03d}_slice.png")
        full_slice_path = os.path.join(full_dose_dir, f"full_3d_{v:03d}_slice.png")
        
        low_slice_img.save(low_slice_path)
        full_slice_img.save(full_slice_path)
    
    print(f"3D样本已保存到 {config.data_dir}")


def test_pipeline(config: DataConfig):
    """测试整个数据管道"""
    print("测试数据管道...")
    
    # 创建虚拟数据
    create_dummy_data(config, num_samples=10)
    
    # 测试数据加载器
    from Module.Loader.data_loader import prepare_data_loaders
    
    try:
        train_loader, val_loader, test_loader = prepare_data_loaders(config)
        
        # 检查批次
        for low_dose, full_dose in train_loader:
            print(f"批次形状 - 低剂量: {low_dose.shape}, 全剂量: {full_dose.shape}")
            print(f"值范围 - 低剂量: [{low_dose.min():.3f}, {low_dose.max():.3f}], "
                  f"全剂量: [{full_dose.min():.3f}, {full_dose.max():.3f}]")
            break
        
        print("数据管道测试通过！")
        return True
        
    except Exception as e:
        print(f"数据管道测试失败: {e}")
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='创建示例CT数据')
    parser.add_argument('--type', type=str, default='2d', choices=['2d', '3d', 'dummy', 'test'],
                       help='数据类型: 2d, 3d, dummy, test')
    parser.add_argument('--num', type=int, default=20,
                       help='样本数量')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据目录')
    
    args = parser.parse_args()
    
    # 配置
    config = DataConfig(data_dir=args.data_dir)
    
    if args.type == '2d':
        create_2d_ct_samples(config, args.num)
    
    elif args.type == '3d':
        create_3d_ct_samples(config, args.num // 5, slices_per_volume=32)
    
    elif args.type == 'dummy':
        create_dummy_data(config, args.num)
    
    elif args.type == 'test':
        test_pipeline(config)
    
    print("完成！")


if __name__ == "__main__":
    main()
