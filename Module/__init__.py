"""
Low-dose CT Image Enhancement AI Module

This module provides tools for enhancing low-dose CT images using deep learning.
"""

__version__ = "1.0.0"
__author__ = "CT Enhancement AI Team"
__license__ = "MIT"

# Core components
from .Config.config import Config, DataConfig, ModelConfig, TrainingConfig, DiagnosticsConfig
from .Loader.data_loader import CTDataset, create_dummy_data, prepare_data_loaders
from .Model.models import UNet2DModel, WaveletDomainCNNModel, FBPConvNetModel, MultiScaleModel
from .Model.losses import SSIMLoss, MixedLoss, MultiScaleLoss, PerceptualLoss
from torch.nn import L1Loss, MSELoss
from .Model.train import Trainer
from .Inference.inference import CTEnhancer
from .Tools.utils import save_checkpoint, load_checkpoint, calculate_metrics, visualize_results
from .Tools.diagnostics import ModelDiagnostics, DiagnosticsCLI
from .Tools.wavelet_transform import DWT2d, LearnableDirectionalWavelet, WaveletDomainProcessing

# Define public API
__all__ = [
    # Config
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "DiagnosticsConfig",
    
    # Data
    "CTDataset",
    "create_dummy_data",
    "prepare_data_loaders",
    
    # Models
    "UNet2DModel",
    "WaveletDomainCNNModel",
    "FBPConvNetModel",
    "MultiScaleModel",
    
    # Losses
    "L1Loss",
    "MSELoss",
    "SSIMLoss",
    "MixedLoss",
    "MultiScaleLoss",
    "PerceptualLoss",
    
    # Training
    "Trainer",
    
    # Inference
    "CTEnhancer",
    
    # Tools
    "save_checkpoint",
    "load_checkpoint",
    "calculate_metrics",
    "visualize_results",
    "ModelDiagnostics",
    "DiagnosticsCLI",
    "DWT2d",
    "LearnableDirectionalWavelet",
    "WaveletDomainProcessing",
    
    # Metadata
    "__version__",
    "__author__",
    "__license__",
]

# Package initialization
print(f"CT Enhancement AI Module v{__version__} loaded successfully.")
print(f"Available components: {len(__all__) - 3} classes/functions")
