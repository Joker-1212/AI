# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Low-dose CT image enhancement framework built with PyTorch and MONAI. Trains deep learning models to denoise low-dose CT images to approach full-dose quality. Documentation and comments are in Chinese.

## Commands

### CLI Entry Point
```bash
python scripts/main.py <command>
```

### Training
```bash
python scripts/main.py train                                          # default config
python scripts/main.py train --config configs/advanced_training_config.yaml
python scripts/main.py train --resume ./checkpoints/checkpoint_epoch_20.pth
```

### Creating Synthetic Training Data
```bash
python scripts/main.py create-data --type 2d --num 20
python scripts/main.py create-data --type 3d --num 5
```

### Inference
```bash
python scripts/main.py enhance --checkpoint models/checkpoints/best_model.pth --input image.nii
```

### Diagnostics
```bash
python -m Module.Tools.diagnostics comprehensive --model best_model.pth --data ./data
```

### Testing & Linting
```bash
pytest                    # run all tests
pytest -x                 # stop on first failure
black .                   # format code
flake8                    # lint
mypy Module/              # type checking
```

## Architecture

The codebase follows a modular design under `Module/`:

- **Config/** (`config.py`) — Dataclasses: `DataConfig`, `ModelConfig`, `TrainingConfig`, `DiagnosticsConfig`. YAML-driven with CLI override support.
- **Loader/** — `CTDataset` handles NIfTI/DICOM/PNG/NumPy input. Data is split 80/10/10 train/val/test. Normalization range is `[-0.1, 0.1]`. Separate optimized loaders exist for Windows.
- **Model/** — `models.py` defines 7 architectures (WaveletDomainCNN, UNet2D/3D, AttentionUNet, ResUNet, DenseUNet, MultiScaleModel, FBPConvNet). `train.py` contains the `Trainer` class. `losses.py` has custom losses (L1, MSE, SSIM, MixedLoss, MultiScaleLoss, PerceptualLoss).
- **Inference/** (`inference.py`) — `CTEnhancer` class for loading checkpoints and enhancing images.
- **Tools/** — Utilities for device management, AMP/mixed-precision, memory optimization, wavelet transforms, and performance monitoring. `diagnostics/` is a modular sub-package for metrics (PSNR, SSIM, LPIPS), visualization, model analysis, and training curve evaluation.

`scripts/main.py` is the CLI entry point that routes to training, data creation, and enhancement commands.

## Key Design Patterns

- **Configuration-driven**: All hyperparameters flow from YAML configs → dataclasses. Three preset configs exist in `configs/` (advanced, simple_optimized, fast).
- **Training features**: Mixed precision (AMP), gradient clipping/accumulation, 6 LR scheduler types (ReduceLROnPlateau, CosineAnnealing, CosineWarmRestarts, StepLR, MultiStepLR, ExponentialLR), warmup epochs, early stopping, checkpoint resume.
- **Device abstraction**: `DeviceManager` auto-detects CUDA/MPS/CPU. Code must work across all three.
- **Data directory convention**: Low-dose images in `qd/` subdirectory, full-dose in `fd/` subdirectory under the data root.

## Dependencies

Core: PyTorch 2.9.1, MONAI 1.5.1, nibabel, pydicom, PyWavelets. See `requirements.txt` for full list and `requirements-dev.txt` for dev tools.
