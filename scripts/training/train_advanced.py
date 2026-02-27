#!/usr/bin/env python3
"""
改进的低剂量CT增强AI训练脚本
支持混合损失函数、梯度裁剪、早停、学习率热身等高级训练策略
"""
import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

# 添加Module目录到路径
sys.path.append(str(Path(__file__).parent))

from Module.Config.config import Config, TrainingConfig, ModelConfig, DataConfig
from Module.Model.train import Trainer
from Module.Loader.data_loader import create_dummy_data
from Module.Tools.utils import save_config


def create_advanced_config(args) -> Config:
    """根据命令行参数创建高级配置"""
    # 加载基础配置
    if args.config and os.path.exists(args.config):
        from Module.Tools.utils import load_config
        config = load_config(args.config, Config)
    else:
        config = Config()
    
    # 更新训练配置
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.optimizer:
        config.training.optimizer = args.optimizer
    if args.loss:
        config.training.loss_function = args.loss
    if args.scheduler:
        config.training.scheduler = args.scheduler
    
    # 高级训练参数
    if args.weight_decay:
        config.training.weight_decay = args.weight_decay
    if args.gradient_clip:
        setattr(config.training, 'gradient_clip_value', args.gradient_clip)
    if args.warmup_epochs:
        setattr(config.training, 'warmup_epochs', args.warmup_epochs)
    if args.patience:
        config.training.patience = args.patience
    
    # 损失函数权重
    if args.loss_weights:
        weights = tuple(map(float, args.loss_weights.split(',')))
        config.training.loss_weights = weights
    
    # 多尺度损失
    if args.multi_scale:
        config.training.use_multi_scale_loss = True
        if args.multi_scale_weights:
            weights = tuple(map(float, args.multi_scale_weights.split(',')))
            config.training.multi_scale_weights = weights
    
    # 模型配置
    if args.model:
        config.model.model_name = args.model
    if args.features:
        features = tuple(map(int, args.features.split(',')))
        config.model.features = features
    
    # 数据配置
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.image_size:
        size = tuple(map(int, args.image_size.split(',')))
        if len(size) == 2:
            config.data.image_size = (size[0], size[1], 1)
        elif len(size) == 3:
            config.data.image_size = size
    
    return config


def train_with_advanced_config(config: Config, args):
    """使用高级配置进行训练"""
    print("=" * 60)
    print("低剂量CT增强AI - 高级训练模式")
    print("=" * 60)
    
    # 打印配置摘要
    print("\n配置摘要:")
    print(f"  模型: {config.model.model_name}")
    print(f"  训练轮数: {config.training.num_epochs}")
    print(f"  批量大小: {config.data.batch_size}")
    print(f"  学习率: {config.training.learning_rate}")
    print(f"  优化器: {config.training.optimizer}")
    print(f"  损失函数: {config.training.loss_function}")
    
    if hasattr(config.training, 'warmup_epochs'):
        print(f"  学习率热身轮数: {config.training.warmup_epochs}")
    
    if hasattr(config.training, 'gradient_clip_value'):
        print(f"  梯度裁剪值: {config.training.gradient_clip_value}")
    
    print(f"  早停耐心值: {config.training.patience}")
    print(f"  检查点目录: {config.training.checkpoint_dir}")
    print(f"  日志目录: {config.training.log_dir}")
    
    # 检查数据
    data_exists = os.path.exists(config.data.data_dir) and \
                  os.listdir(os.path.join(config.data.data_dir, config.data.low_dose_dir))
    
    if not data_exists:
        print("\n未找到数据，创建虚拟数据...")
        create_dummy_data(config.data, num_samples=args.samples if args.samples else 50)
    
    # 保存配置
    config_save_path = os.path.join(config.training.checkpoint_dir, "config.yaml")
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    save_config(config, config_save_path)
    print(f"\n配置已保存到: {config_save_path}")
    
    # 创建训练器
    print("\n初始化训练器...")
    
    # 检查是否从检查点恢复
    checkpoint_path = None
    
    # 优先级1: 命令行参数 --resume
    if hasattr(args, 'resume') and args.resume:
        checkpoint_path = args.resume
    
    # 优先级2: 配置文件中的 resume_checkpoint
    if not checkpoint_path and hasattr(config.training, 'resume_checkpoint'):
        checkpoint_path = config.training.resume_checkpoint
    
    # 验证检查点路径
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            print(f"警告: 检查点文件不存在: {checkpoint_path}")
            print("将从头开始训练...")
            checkpoint_path = None
        else:
            print(f"从检查点恢复: {checkpoint_path}")
    
    trainer = Trainer(config, checkpoint_path=checkpoint_path)
    
    # 训练
    print("\n开始训练...")
    train_losses, val_losses, val_psnrs, val_ssims = trainer.train()
    
    # 测试
    print("\n在测试集上评估...")
    test_loss, test_psnr, test_ssim = trainer.test()
    
    # 保存最终结果
    results = {
        'best_val_loss': trainer.best_val_loss,
        'best_val_psnr': max(val_psnrs) if val_psnrs else 0,
        'best_val_ssim': max(val_ssims) if val_ssims else 0,
        'test_loss': test_loss,
        'test_psnr': test_psnr,
        'test_ssim': test_ssim,
        'total_epochs': len(train_losses)
    }
    
    results_path = os.path.join(config.training.checkpoint_dir, "results.yaml")
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    print(f"\n训练结果已保存到: {results_path}")
    print("\n训练完成！")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="改进的低剂量CT增强AI训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基本参数
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--data-dir", type=str, help="数据目录路径")
    parser.add_argument("--samples", type=int, default=50, help="虚拟数据样本数（如果无数据）")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--batch-size", type=int, help="批量大小")
    parser.add_argument("--learning-rate", type=float, help="学习率")
    parser.add_argument("--optimizer", type=str, 
                       choices=["Adam", "AdamW", "SGD"], help="优化器")
    parser.add_argument("--loss", type=str, 
                       choices=["L1Loss", "MSELoss", "SSIMLoss", "MixedLoss", "MultiScaleLoss"],
                       help="损失函数")
    parser.add_argument("--scheduler", type=str,
                       choices=["ReduceLROnPlateau", "Cosine", "Step", "MultiStep", "CosineWarmRestarts"],
                       help="学习率调度器")
    
    # 高级训练参数
    parser.add_argument("--weight-decay", type=float, help="权重衰减")
    parser.add_argument("--gradient-clip", type=float, help="梯度裁剪值")
    parser.add_argument("--warmup-epochs", type=int, help="学习率热身轮数")
    parser.add_argument("--patience", type=int, help="早停耐心值")
    
    # 损失函数参数
    parser.add_argument("--loss-weights", type=str, 
                       help="混合损失权重（逗号分隔，如：1.0,0.5,0.1）")
    parser.add_argument("--multi-scale", action="store_true",
                       help="使用多尺度损失")
    parser.add_argument("--multi-scale-weights", type=str,
                       help="多尺度损失权重（逗号分隔，如：1.0,0.5,0.25）")
    
    # 模型参数
    parser.add_argument("--model", type=str,
                       choices=["UNet2D", "WaveletDomainCNN", "FBPConvNet", "MultiScaleModel"],
                       help="模型类型")
    parser.add_argument("--features", type=str,
                       help="特征通道数（逗号分隔，如：32,64,128,256）")
    
    # 数据参数
    parser.add_argument("--image-size", type=str,
                       help="图像尺寸（逗号分隔，如：512,512 或 512,512,32）")
    
    # 实验管理
    parser.add_argument("--experiment", type=str, help="实验名称")
    parser.add_argument("--resume", type=str, help="从检查点恢复训练")
    
    args = parser.parse_args()
    
    # 如果没有参数，显示帮助
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # 创建配置
    config = create_advanced_config(args)
    
    # 应用实验名称
    if args.experiment:
        config.training.checkpoint_dir = os.path.join(
            config.training.checkpoint_dir, args.experiment
        )
        config.training.log_dir = os.path.join(
            config.training.log_dir, args.experiment
        )
    
    # 训练
    results = train_with_advanced_config(config, args)
    
    # 打印最终结果
    print("\n" + "=" * 60)
    print("最终结果:")
    print("=" * 60)
    print(f"  最佳验证损失: {results['best_val_loss']:.6f}")
    print(f"  最佳验证PSNR: {results['best_val_psnr']:.2f} dB")
    print(f"  最佳验证SSIM: {results['best_val_ssim']:.4f}")
    print(f"  测试损失: {results['test_loss']:.6f}")
    print(f"  测试PSNR: {results['test_psnr']:.2f} dB")
    print(f"  测试SSIM: {results['test_ssim']:.4f}")
    print(f"  总训练轮数: {results['total_epochs']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
