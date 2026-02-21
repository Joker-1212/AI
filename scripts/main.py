#!/usr/bin/env python3
"""
Low-dose CT Enhancement AI - Main Entry Point
"""

import os
import sys
import argparse
from pathlib import Path

# Add scripts directory to path for utilities
sys.path.append(str(Path(__file__).parent))

from scripts.utils.logging import get_script_logger
from scripts.utils.error_handler import handle_exceptions, ScriptError
from scripts.utils.arg_parser import create_subcommand_parser

from Module.Config.config import Config
from Module.Loader.data_loader import create_dummy_data
from Module.Model.train import Trainer
from Module.Inference.inference import CTEnhancer

# Setup logger
logger = get_script_logger("main")


@handle_exceptions(log_error=True, exit_on_error=True)
def train_model(config_path=None, verbose=False):
    """Train model with optional configuration file."""
    logger.info("Starting model training...")
    
    if config_path and os.path.exists(config_path):
        from Module.Tools.utils import load_config
        config = load_config(config_path, Config)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        config = Config()
        logger.info("Using default configuration")
    
    # Check data
    data_exists = os.path.exists(config.data.data_dir) and \
                  os.listdir(os.path.join(config.data.data_dir, config.data.low_dose_dir))
    
    if not data_exists:
        logger.info("No data found, creating dummy data...")
        create_dummy_data(config.data, num_samples=50)
    
    # Train
    logger.info("Initializing trainer...")
    trainer = Trainer(config)
    
    logger.info("Starting training process...")
    trainer.train()
    
    logger.info("Running evaluation on test set...")
    trainer.test()
    
    logger.info("Training completed successfully!")


@handle_exceptions(log_error=True, exit_on_error=True)
def enhance_image(checkpoint_path, input_path, output_path=None):
    """Enhance single image."""
    logger.info(f"Enhancing image: {input_path}")
    
    enhancer = CTEnhancer(checkpoint_path)
    output = enhancer.enhance_file(input_path, output_path)
    
    logger.info(f"Enhanced image saved to: {output}")
    return output


@handle_exceptions(log_error=True, exit_on_error=True)
def enhance_directory(checkpoint_path, input_dir, output_dir=None, pattern="*.nii"):
    """Enhance all images in directory."""
    if output_dir is None:
        output_dir = os.path.join(input_dir, "enhanced")
    
    logger.info(f"Enhancing directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"File pattern: {pattern}")
    
    enhancer = CTEnhancer(checkpoint_path)
    outputs = enhancer.enhance_directory(input_dir, output_dir, pattern)
    
    logger.info(f"Enhancement completed, processed {len(outputs)} files")
    return outputs


@handle_exceptions(log_error=True, exit_on_error=True)
def create_sample_data(data_type="2d", num_samples=20):
    """Create sample data for testing."""
    logger.info(f"Creating {data_type} sample data ({num_samples} samples)")
    
    from scripts.data.create_sample_data import create_2d_ct_samples, create_3d_ct_samples
    
    config = Config().data
    
    if data_type == "2d":
        create_2d_ct_samples(config, num_samples)
    elif data_type == "3d":
        create_3d_ct_samples(config, num_samples // 5)
    else:
        create_dummy_data(config, num_samples)
    
    logger.info(f"Sample data created at {config.data_dir}")


@handle_exceptions(log_error=True, exit_on_error=True)
def test_pipeline():
    """Test the entire pipeline."""
    logger.info("Testing pipeline...")
    
    from scripts.data.create_sample_data import test_pipeline as test_data_pipeline
    config = Config().data
    success = test_data_pipeline(config)
    
    if success:
        logger.info("Pipeline test passed!")
    else:
        logger.error("Pipeline test failed!")
        raise ScriptError("Pipeline test failed")


def main():
    """Main entry point."""
    parser = create_subcommand_parser()
    args = parser.parse_args()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Configure logging level based on arguments
    if hasattr(args, 'verbose') and args.verbose:
        logger.setLevel("DEBUG")
        logger.debug("Verbose mode enabled")
    
    logger.info(f"Executing command: {args.command}")
    
    # Execute command
    if args.command == "train":
        train_model(args.config, args.verbose if hasattr(args, 'verbose') else False)
    
    elif args.command == "enhance":
        if os.path.isfile(args.input):
            enhance_image(args.checkpoint, args.input, args.output)
        elif os.path.isdir(args.input):
            enhance_directory(args.checkpoint, args.input, args.output, args.pattern)
        else:
            logger.error(f"Input path does not exist: {args.input}")
            sys.exit(1)
    
    elif args.command == "create-data":
        create_sample_data(args.type, args.num)
    
    elif args.command == "test":
        test_pipeline()
    
    else:
        parser.print_help()
    
    logger.info("Command execution completed")


if __name__ == "__main__":
    main()
