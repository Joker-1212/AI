#!/usr/bin/env python3
"""
Command-line argument parsing utilities.
"""

import argparse
import sys
from typing import List, Optional, Dict, Any


def create_base_parser(description: str = "CT Enhancement Script") -> argparse.ArgumentParser:
    """
    Create base argument parser with common options.
    
    Args:
        description: Script description
    
    Returns:
        Argument parser
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Common options
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for log files"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without actual execution"
    )
    
    return parser


def add_data_args(parser: argparse.ArgumentParser) -> None:
    """Add data-related arguments to parser."""
    data_group = parser.add_argument_group("Data options")
    
    data_group.add_argument(
        "--data-dir",
        type=str,
        help="Data directory path"
    )
    
    data_group.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training/inference"
    )
    
    data_group.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )
    
    data_group.add_argument(
        "--image-size",
        type=str,
        help="Image size (comma-separated, e.g., '512,512' or '512,512,32')"
    )


def add_training_args(parser: argparse.ArgumentParser) -> None:
    """Add training-related arguments to parser."""
    train_group = parser.add_argument_group("Training options")
    
    train_group.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs"
    )
    
    train_group.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate"
    )
    
    train_group.add_argument(
        "--optimizer",
        type=str,
        choices=["Adam", "AdamW", "SGD"],
        help="Optimizer"
    )
    
    train_group.add_argument(
        "--loss",
        type=str,
        choices=["L1Loss", "MSELoss", "SSIMLoss", "MixedLoss", "MultiScaleLoss"],
        help="Loss function"
    )
    
    train_group.add_argument(
        "--scheduler",
        type=str,
        choices=["ReduceLROnPlateau", "Cosine", "Step", "MultiStep", "CosineWarmRestarts"],
        help="Learning rate scheduler"
    )
    
    train_group.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for model checkpoints"
    )
    
    train_group.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume training from"
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add model-related arguments to parser."""
    model_group = parser.add_argument_group("Model options")
    
    model_group.add_argument(
        "--model",
        type=str,
        choices=["UNet2D", "WaveletDomainCNN", "FBPConvNet", "MultiScaleModel"],
        help="Model architecture"
    )
    
    model_group.add_argument(
        "--features",
        type=str,
        help="Feature channels (comma-separated, e.g., '32,64,128,256')"
    )
    
    model_group.add_argument(
        "--pretrained",
        type=str,
        help="Path to pretrained model weights"
    )


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    """Add inference-related arguments to parser."""
    infer_group = parser.add_argument_group("Inference options")
    
    infer_group.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    infer_group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image path or directory"
    )
    
    infer_group.add_argument(
        "--output",
        type=str,
        help="Output path"
    )
    
    infer_group.add_argument(
        "--pattern",
        type=str,
        default="*.nii",
        help="File pattern for directory inference"
    )
    
    infer_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference"
    )


def parse_image_size(image_size_str: str) -> tuple:
    """
    Parse image size string to tuple.
    
    Args:
        image_size_str: Comma-separated image size string
    
    Returns:
        Tuple of integers
    """
    if not image_size_str:
        return None
    
    try:
        sizes = list(map(int, image_size_str.split(',')))
        return tuple(sizes)
    except ValueError:
        raise ValueError(f"Invalid image size format: {image_size_str}")


def parse_features(features_str: str) -> tuple:
    """
    Parse features string to tuple.
    
    Args:
        features_str: Comma-separated features string
    
    Returns:
        Tuple of integers
    """
    if not features_str:
        return None
    
    try:
        features = list(map(int, features_str.split(',')))
        return tuple(features)
    except ValueError:
        raise ValueError(f"Invalid features format: {features_str}")


def parse_loss_weights(weights_str: str) -> tuple:
    """
    Parse loss weights string to tuple.
    
    Args:
        weights_str: Comma-separated weights string
    
    Returns:
        Tuple of floats
    """
    if not weights_str:
        return None
    
    try:
        weights = list(map(float, weights_str.split(',')))
        return tuple(weights)
    except ValueError:
        raise ValueError(f"Invalid weights format: {weights_str}")


def create_subcommand_parser() -> argparse.ArgumentParser:
    """
    Create parser with subcommands for different operations.
    
    Returns:
        Argument parser with subcommands
    """
    parser = argparse.ArgumentParser(
        description="CT Enhancement AI - Command Line Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    add_training_args(train_parser)
    add_data_args(train_parser)
    add_model_args(train_parser)
    train_parser.add_argument("--config", type=str, help="Configuration file")
    
    # Enhance command
    enhance_parser = subparsers.add_parser("enhance", help="Enhance images")
    add_inference_args(enhance_parser)
    
    # Create-data command
    data_parser = subparsers.add_parser("create-data", help="Create sample data")
    data_parser.add_argument("--type", type=str, default="2d", 
                           choices=["2d", "3d", "dummy"],
                           help="Data type")
    data_parser.add_argument("--num", type=int, default=20,
                           help="Number of samples")
    data_parser.add_argument("--data-dir", type=str, default="./data",
                           help="Data directory")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test pipeline")
    test_parser.add_argument("--config", type=str, help="Configuration file")
    
    return parser


# Import torch for device detection
try:
    import torch
except ImportError:
    torch = None
