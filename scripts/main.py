#!/usr/bin/env python3
"""
Low-dose CT Enhancement AI - Main Entry Point

This script provides the main entry point for the CT enhancement AI pipeline.
It handles proper module imports and path management.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path for proper module imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now we can import modules properly
try:
    from Module.Config.config import Config
    from Module.Loader.data_loader import create_dummy_data
    from Module.Model.train import Trainer
    from Module.Inference.inference import CTEnhancer
    from scripts.utils.logging import get_script_logger
    from scripts.utils.error_handler import handle_exceptions, ScriptError
    from scripts.utils.arg_parser import create_subcommand_parser
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you are running from the project root directory.")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    sys.exit(1)

# Setup logger
logger = get_script_logger("main")


def _apply_config_overrides(config, overrides):
    """
    Apply command-line overrides to configuration.
    
    Args:
        config: Configuration object
        overrides: Dictionary of override parameters
        
    Returns:
        Updated configuration
    """
    # Training parameters
    if 'epochs' in overrides and overrides['epochs']:
        config.training.num_epochs = overrides['epochs']
    
    if 'learning_rate' in overrides and overrides['learning_rate']:
        config.training.learning_rate = overrides['learning_rate']
    
    if 'optimizer' in overrides and overrides['optimizer']:
        config.training.optimizer = overrides['optimizer']
    
    if 'loss' in overrides and overrides['loss']:
        config.training.loss_function = overrides['loss']
    
    if 'scheduler' in overrides and overrides['scheduler']:
        config.training.scheduler = overrides['scheduler']
    
    if 'checkpoint_dir' in overrides and overrides['checkpoint_dir']:
        config.training.checkpoint_dir = overrides['checkpoint_dir']
    
    # Advanced training parameters
    if 'weight_decay' in overrides and overrides['weight_decay']:
        config.training.weight_decay = overrides['weight_decay']
    
    if 'gradient_clip' in overrides and overrides['gradient_clip']:
        setattr(config.training, 'gradient_clip_value', overrides['gradient_clip'])
    
    if 'warmup_epochs' in overrides and overrides['warmup_epochs']:
        setattr(config.training, 'warmup_epochs', overrides['warmup_epochs'])
    
    if 'patience' in overrides and overrides['patience']:
        config.training.patience = overrides['patience']
    
    # Loss function parameters
    if 'loss_weights' in overrides and overrides['loss_weights']:
        from scripts.utils.arg_parser import parse_loss_weights
        weights = parse_loss_weights(overrides['loss_weights'])
        if weights:
            config.training.loss_weights = weights
    
    if 'multi_scale' in overrides and overrides['multi_scale']:
        config.training.use_multi_scale_loss = True
    
    if 'multi_scale_weights' in overrides and overrides['multi_scale_weights']:
        from scripts.utils.arg_parser import parse_loss_weights
        weights = parse_loss_weights(overrides['multi_scale_weights'])
        if weights:
            config.training.multi_scale_weights = weights
    
    # Experiment name
    if 'experiment' in overrides and overrides['experiment']:
        # Update checkpoint and log directories with experiment name
        experiment_name = overrides['experiment']
        config.training.checkpoint_dir = os.path.join(
            config.training.checkpoint_dir, experiment_name
        )
        config.training.log_dir = os.path.join(
            config.training.log_dir, experiment_name
        )
    
    # Data parameters
    if 'batch_size' in overrides and overrides['batch_size']:
        config.data.batch_size = overrides['batch_size']
    
    if 'data_dir' in overrides and overrides['data_dir']:
        config.data.data_dir = overrides['data_dir']
    
    if 'image_size' in overrides and overrides['image_size']:
        from scripts.utils.arg_parser import parse_image_size
        size = parse_image_size(overrides['image_size'])
        if size:
            if len(size) == 2:
                config.data.image_size = (size[0], size[1], 1)
            elif len(size) == 3:
                config.data.image_size = size
    
    # Model parameters
    if 'model' in overrides and overrides['model']:
        config.model.model_name = overrides['model']
    
    if 'features' in overrides and overrides['features']:
        from scripts.utils.arg_parser import parse_features
        features = parse_features(overrides['features'])
        if features:
            config.model.features = features
    
    return config


@handle_exceptions(log_error=True, exit_on_error=True)
def train_model(config_path=None, verbose=False, resume=None, **kwargs):
    """
    Train model with optional configuration file and resume support.
    
    Args:
        config_path: Path to configuration file
        verbose: Enable verbose logging
        resume: Path to checkpoint to resume training from
        **kwargs: Additional training parameters to override config
    """
    logger.info("Starting model training...")
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        from Module.Tools.utils import load_config
        config = load_config(config_path, Config)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        config = Config()
        logger.info("Using default configuration")
    
    # Apply command-line overrides to config
    if kwargs:
        config = _apply_config_overrides(config, kwargs)
    
    # Check data
    data_exists = os.path.exists(config.data.data_dir) and \
                  os.listdir(os.path.join(config.data.data_dir, config.data.low_dose_dir))
    
    if not data_exists:
        logger.info("No data found, creating dummy data...")
        create_dummy_data(config.data, num_samples=50)
    
    # Determine checkpoint path for resuming
    checkpoint_path = None
    
    # Priority 1: Command-line --resume argument
    if resume:
        checkpoint_path = resume
    
    # Priority 2: Config file resume_checkpoint
    if not checkpoint_path and hasattr(config.training, 'resume_checkpoint'):
        checkpoint_path = config.training.resume_checkpoint
    
    # Validate checkpoint path
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint file does not exist: {checkpoint_path}")
            logger.warning("Starting training from scratch...")
            checkpoint_path = None
        else:
            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
    
    # Save configuration
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    config_save_path = os.path.join(config.training.checkpoint_dir, "config.yaml")
    from Module.Tools.utils import save_config
    save_config(config, config_save_path)
    logger.info(f"Configuration saved to: {config_save_path}")
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(config, checkpoint_path=checkpoint_path)
    
    # Train
    logger.info("Starting training process...")
    train_losses, val_losses, val_psnrs, val_ssims = trainer.train()
    
    # Test
    logger.info("Running evaluation on test set...")
    test_loss, test_psnr, test_ssim = trainer.test()
    
    # Save results
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
    import yaml
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    logger.info(f"Training results saved to: {results_path}")
    logger.info("Training completed successfully!")
    
    return results


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
        # Collect all training-related arguments
        train_kwargs = {}
        
        # Training parameters
        if hasattr(args, 'epochs') and args.epochs:
            train_kwargs['epochs'] = args.epochs
        
        if hasattr(args, 'learning_rate') and args.learning_rate:
            train_kwargs['learning_rate'] = args.learning_rate
        
        if hasattr(args, 'optimizer') and args.optimizer:
            train_kwargs['optimizer'] = args.optimizer
        
        if hasattr(args, 'loss') and args.loss:
            train_kwargs['loss'] = args.loss
        
        if hasattr(args, 'scheduler') and args.scheduler:
            train_kwargs['scheduler'] = args.scheduler
        
        if hasattr(args, 'checkpoint_dir') and args.checkpoint_dir:
            train_kwargs['checkpoint_dir'] = args.checkpoint_dir
        
        if hasattr(args, 'resume') and args.resume:
            train_kwargs['resume'] = args.resume
        
        # Advanced training parameters
        if hasattr(args, 'weight_decay') and args.weight_decay:
            train_kwargs['weight_decay'] = args.weight_decay
        
        if hasattr(args, 'gradient_clip') and args.gradient_clip:
            train_kwargs['gradient_clip'] = args.gradient_clip
        
        if hasattr(args, 'warmup_epochs') and args.warmup_epochs:
            train_kwargs['warmup_epochs'] = args.warmup_epochs
        
        if hasattr(args, 'patience') and args.patience:
            train_kwargs['patience'] = args.patience
        
        # Loss function parameters
        if hasattr(args, 'loss_weights') and args.loss_weights:
            train_kwargs['loss_weights'] = args.loss_weights
        
        if hasattr(args, 'multi_scale') and args.multi_scale:
            train_kwargs['multi_scale'] = args.multi_scale
        
        if hasattr(args, 'multi_scale_weights') and args.multi_scale_weights:
            train_kwargs['multi_scale_weights'] = args.multi_scale_weights
        
        # Experiment management
        if hasattr(args, 'experiment') and args.experiment:
            train_kwargs['experiment'] = args.experiment
        
        # Data parameters
        if hasattr(args, 'batch_size') and args.batch_size:
            train_kwargs['batch_size'] = args.batch_size
        
        if hasattr(args, 'data_dir') and args.data_dir:
            train_kwargs['data_dir'] = args.data_dir
        
        if hasattr(args, 'image_size') and args.image_size:
            train_kwargs['image_size'] = args.image_size
        
        # Model parameters
        if hasattr(args, 'model') and args.model:
            train_kwargs['model'] = args.model
        
        if hasattr(args, 'features') and args.features:
            train_kwargs['features'] = args.features
        
        train_model(
            config_path=args.config,
            verbose=args.verbose if hasattr(args, 'verbose') else False,
            **train_kwargs
        )
    
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
