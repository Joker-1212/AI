#!/usr/bin/env python3
"""
低剂量CT增强AI - 主入口点
"""
import os
import sys
import argparse
from pathlib import Path

# 添加src目录到路径
# sys.path.append(str(Path(__file__).parent / "src"))

from Module.Config.config import Config
from Module.Loader.data_loader import create_dummy_data
from Module.Models.train import Trainer
from Module.Inference.inference import CTEnhancer


def train_model(config_path=None):
    """训练模型"""
    if config_path and os.path.exists(config_path):
        from Module.Tools.utils import load_config
        config = load_config(config_path, Config)
    else:
        config = Config()
    
    # 如果没有数据，创建虚拟数据
    data_exists = os.path.exists(config.data.data_dir) and \
                  os.listdir(os.path.join(config.data.data_dir, config.data.low_dose_dir))
    
    if not data_exists:
        print("未找到数据，创建虚拟数据...")
        create_dummy_data(config.data, num_samples=50)
    
    # 训练
    trainer = Trainer(config)
    trainer.train()
    
    # 测试
    trainer.test()
    
    print("训练完成！")


def enhance_image(checkpoint_path, input_path, output_path=None):
    """增强单张图像"""
    enhancer = CTEnhancer(checkpoint_path)
    output = enhancer.enhance_file(input_path, output_path)
    print(f"增强图像已保存到: {output}")
    return output


def enhance_directory(checkpoint_path, input_dir, output_dir=None, pattern="*.nii"):
    """增强目录中的所有图像"""
    if output_dir is None:
        output_dir = os.path.join(input_dir, "enhanced")
    
    enhancer = CTEnhancer(checkpoint_path)
    outputs = enhancer.enhance_directory(input_dir, output_dir, pattern)
    print(f"增强完成，共处理 {len(outputs)} 个文件")
    return outputs


def create_sample_data(data_type="2d", num_samples=20):
    """创建示例数据"""
    from scripts.create_sample_data import create_2d_ct_samples, create_3d_ct_samples
    
    config = Config().data
    
    if data_type == "2d":
        create_2d_ct_samples(config, num_samples)
    elif data_type == "3d":
        create_3d_ct_samples(config, num_samples // 5)
    else:
        create_dummy_data(config, num_samples)
    
    print(f"示例数据已创建到 {config.data_dir}")


def main():
    parser = argparse.ArgumentParser(description="低剂量CT增强AI")
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--config", type=str, help="配置文件路径")
    
    # 增强命令
    enhance_parser = subparsers.add_parser("enhance", help="增强图像")
    enhance_parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    enhance_parser.add_argument("--input", type=str, required=True, help="输入图像路径或目录")
    enhance_parser.add_argument("--output", type=str, help="输出路径")
    enhance_parser.add_argument("--pattern", type=str, default="*.nii", help="文件匹配模式")
    
    # 创建数据命令
    data_parser = subparsers.add_parser("create-data", help="创建示例数据")
    data_parser.add_argument("--type", type=str, default="2d", choices=["2d", "3d", "dummy"],
                            help="数据类型")
    data_parser.add_argument("--num", type=int, default=20, help="样本数量")
    
    # 测试命令
    test_parser = subparsers.add_parser("test", help="测试管道")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_model(args.config)
    
    elif args.command == "enhance":
        if os.path.isfile(args.input):
            enhance_image(args.checkpoint, args.input, args.output)
        elif os.path.isdir(args.input):
            enhance_directory(args.checkpoint, args.input, args.output, args.pattern)
        else:
            print(f"输入路径不存在: {args.input}")
    
    elif args.command == "create-data":
        create_sample_data(args.type, args.num)
    
    elif args.command == "test":
        from scripts.create_sample_data import test_pipeline
        config = Config().data
        test_pipeline(config)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
