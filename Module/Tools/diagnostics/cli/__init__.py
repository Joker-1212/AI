"""
Command Line Interface Module

Provides DiagnosticsCLI class for command-line diagnostic tools.
"""

import argparse
import sys
import os
from typing import Optional, List, Dict, Any

from ..config import DiagnosticsConfig
from ..metrics import ImageMetricsCalculator
from ..visualization import ValidationVisualizer
from ..model import ModelDiagnostics
from ..analysis import TrainingCurveAnalyzer


class DiagnosticsCLI:
    """
    Command-line interface for diagnostic tools
    
    Provides various command-line utilities for:
    - Calculating image metrics
    - Visualizing validation results
    - Analyzing model diagnostics
    - Generating training reports
    """
    
    def __init__(self, config: Optional[DiagnosticsConfig] = None):
        """
        Initialize CLI tool
        
        Args:
            config: Diagnostics configuration, uses default if None
        """
        self.config = config or DiagnosticsConfig()
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description='Low-dose CT Enhancement AI Diagnostic Tools',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Calculate metrics for two images
  python -m Module.Tools.diagnostics.cli calculate --pred pred.npy --target target.npy
  
  # Visualize validation results
  python -m Module.Tools.diagnostics.cli visualize --low low.npy --enhanced enhanced.npy --target target.npy
  
  # Analyze model weights
  python -m Module.Tools.diagnostics.cli analyze-model --model checkpoint.pth
            """
        )
        
        # Subparsers for different commands
        subparsers = parser.add_subparsers(dest='command', help='Command to execute')
        
        # Calculate command
        calc_parser = subparsers.add_parser('calculate', help='Calculate image metrics')
        calc_parser.add_argument('--pred', required=True, help='Prediction image file')
        calc_parser.add_argument('--target', required=True, help='Target image file')
        calc_parser.add_argument('--data-range', type=float, help='Data range for PSNR')
        calc_parser.add_argument('--output', '-o', help='Output JSON file for metrics')
        
        # Visualize command
        viz_parser = subparsers.add_parser('visualize', help='Visualize validation results')
        viz_parser.add_argument('--low', required=True, help='Low-dose input image')
        viz_parser.add_argument('--enhanced', required=True, help='Enhanced output image')
        viz_parser.add_argument('--target', required=True, help='Full-dose target image')
        viz_parser.add_argument('--output', '-o', required=True, help='Output image file')
        viz_parser.add_argument('--sample-idx', type=int, default=0, help='Sample index')
        
        # Analyze model command
        model_parser = subparsers.add_parser('analyze-model', help='Analyze model diagnostics')
        model_parser.add_argument('--model', required=True, help='Model checkpoint file')
        model_parser.add_argument('--check-weights', action='store_true', help='Check weight distribution')
        model_parser.add_argument('--check-gradients', action='store_true', help='Check gradient issues')
        model_parser.add_argument('--output', '-o', help='Output report file')
        
        # Training analysis command
        train_parser = subparsers.add_parser('analyze-training', help='Analyze training curves')
        train_parser.add_argument('--train-loss', required=True, help='Training losses CSV file')
        train_parser.add_argument('--val-loss', required=True, help='Validation losses CSV file')
        train_parser.add_argument('--output', '-o', help='Output report file')
        
        return parser
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Run CLI with arguments
        
        Args:
            args: Command-line arguments, uses sys.argv if None
            
        Returns:
            Exit code (0 for success, 1 for error)
        """
        if args is None:
            args = sys.argv[1:]
        
        if not args:
            self.parser.print_help()
            return 0
        
        parsed_args = self.parser.parse_args(args)
        
        try:
            if parsed_args.command == 'calculate':
                return self._run_calculate(parsed_args)
            elif parsed_args.command == 'visualize':
                return self._run_visualize(parsed_args)
            elif parsed_args.command == 'analyze-model':
                return self._run_analyze_model(parsed_args)
            elif parsed_args.command == 'analyze-training':
                return self._run_analyze_training(parsed_args)
            else:
                print(f"Unknown command: {parsed_args.command}")
                return 1
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    def _run_calculate(self, args) -> int:
        """Run calculate command"""
        import torch
        import numpy as np
        import json
        
        # Load images
        pred = self._load_image(args.pred)
        target = self._load_image(args.target)
        
        # Calculate metrics
        calculator = ImageMetricsCalculator(self.config)
        metrics = calculator.calculate_all_metrics(pred, target, args.data_range)
        
        # Print results
        print("Image Metrics:")
        print("-" * 40)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:15s}: {value:.4f}")
            elif isinstance(value, (int, str)):
                print(f"{key:15s}: {value}")
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\nMetrics saved to: {args.output}")
        
        return 0
    
    def _run_visualize(self, args) -> int:
        """Run visualize command"""
        import torch
        
        # Load images
        low = self._load_image(args.low)
        enhanced = self._load_image(args.enhanced)
        target = self._load_image(args.target)
        
        # Create visualization
        visualizer = ValidationVisualizer(self.config)
        fig = visualizer.visualize_sample(
            low, enhanced, target,
            sample_idx=args.sample_idx,
            save_path=args.output,
            show=False
        )
        
        print(f"Visualization saved to: {args.output}")
        return 0
    
    def _run_analyze_model(self, args) -> int:
        """Run analyze-model command"""
        import torch
        
        # Load model
        model = self._load_model(args.model)
        
        # Analyze model
        model_diagnostics = ModelDiagnostics(self.config)
        
        if args.check_weights:
            weight_stats = model_diagnostics.analyze_weight_distribution(model)
            print("Weight Distribution Analysis:")
            print("-" * 40)
            print(f"Total parameters: {weight_stats.get('total_params', 0):,}")
            print(f"Trainable parameters: {weight_stats.get('trainable_params', 0):,}")
            
            if 'global_stats' in weight_stats:
                stats = weight_stats['global_stats']
                print(f"Global weight mean: {stats.get('mean', 0):.6f}")
                print(f"Global weight std: {stats.get('std', 0):.6f}")
        
        # Save report if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump({'model_analysis': 'Placeholder'}, f, indent=2)
            print(f"Model analysis saved to: {args.output}")
        
        return 0
    
    def _run_analyze_training(self, args) -> int:
        """Run analyze-training command"""
        import pandas as pd
        
        # Load training data
        train_losses = pd.read_csv(args.train_loss)['loss'].tolist()
        val_losses = pd.read_csv(args.val_loss)['loss'].tolist()
        
        # Analyze training
        analyzer = TrainingCurveAnalyzer(self.config)
        analysis = analyzer.analyze_overfitting(train_losses, val_losses)
        
        # Print results
        print("Training Analysis:")
        print("-" * 40)
        print(f"Training epochs: {len(train_losses)}")
        print(f"Final training loss: {analysis['final_train_loss']:.6f}")
        print(f"Final validation loss: {analysis['final_val_loss']:.6f}")
        print(f"Best validation loss: {analysis['min_val_loss']:.6f} (epoch {analysis['min_val_epoch']})")
        
        if analysis['overfitting_detected']:
            print(f"⚠️  Overfitting detected at epoch {analysis['overfitting_epoch']}")
        else:
            print("✓ No significant overfitting detected")
        
        # Save report if requested
        if args.output:
            report = analyzer.generate_training_report({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epochs': list(range(1, len(train_losses) + 1))
            })
            
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Training report saved to: {args.output}")
        
        return 0
    
    def _load_image(self, filepath: str):
        """Load image from file"""
        import torch
        import numpy as np
        
        if filepath.endswith('.npy'):
            data = np.load(filepath)
        elif filepath.endswith('.pt') or filepath.endswith('.pth'):
            data = torch.load(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        
        return data
    
    def _load_model(self, filepath: str):
        """Load model from checkpoint"""
        import torch
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Simple placeholder - actual implementation would load the specific model architecture
        checkpoint = torch.load(filepath, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            # Create a simple model for demonstration
            class SimpleModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
                    self.conv2 = torch.nn.Conv2d(16, 1, 3, padding=1)
                
                def forward(self, x):
                    return self.conv2(self.conv1(x))
            
            model = SimpleModel()
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        else:
            raise ValueError("Checkpoint does not contain model_state_dict")


def main():
    """Main entry point for CLI"""
    cli = DiagnosticsCLI()
    sys.exit(cli.run())


if __name__ == '__main__':
    main()


__all__ = ['DiagnosticsCLI', 'main']
