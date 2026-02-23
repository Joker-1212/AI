"""
Training Curve Analysis Module

Provides TrainingCurveAnalyzer class for analyzing training loss and metric curves.
"""

import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..config import DiagnosticsConfig


class TrainingCurveAnalyzer:
    """
    Training curve analysis tool
    
    Analyzes:
    - Training/validation loss curves
    - Overfitting detection
    - Learning rate analysis
    - Metric trend analysis
    """
    
    def __init__(self, config: Optional[DiagnosticsConfig] = None):
        """
        Initialize training curve analyzer
        
        Args:
            config: Diagnostics configuration, uses default if None
        """
        self.config = config or DiagnosticsConfig()
    
    def analyze_overfitting(
        self,
        train_losses: List[float],
        val_losses: List[float],
        epochs: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Analyze overfitting phenomenon
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            epochs: Corresponding epoch numbers, uses indices if None
            
        Returns:
            Overfitting analysis results
        """
        if epochs is None:
            epochs = list(range(1, len(train_losses) + 1))
        
        if len(train_losses) != len(val_losses):
            raise ValueError("Training and validation losses must have same length")
        
        # Convert to numpy arrays
        train_arr = np.array(train_losses)
        val_arr = np.array(val_losses)
        
        # Calculate loss ratio
        loss_ratio = val_arr / (train_arr + 1e-8)
        
        # Detect overfitting
        overfitting_detected = False
        overfitting_epoch = -1
        
        # Simple heuristic: if validation loss starts rising while training loss continues to fall
        if len(val_arr) > 10:
            # Calculate recent trends
            recent_val = val_arr[-5:]
            recent_train = train_arr[-5:]
            
            val_trend = np.polyfit(range(5), recent_val, 1)[0]  # slope
            train_trend = np.polyfit(range(5), recent_train, 1)[0]
            
            if val_trend > 0 and train_trend < 0:  # validation rising, training falling
                overfitting_detected = True
                overfitting_epoch = epochs[-5]
        
        # Calculate statistics
        analysis = {
            'epochs': epochs,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'loss_ratio': loss_ratio.tolist(),
            'final_train_loss': float(train_arr[-1]),
            'final_val_loss': float(val_arr[-1]),
            'min_val_loss': float(val_arr.min()),
            'min_val_epoch': int(epochs[val_arr.argmin()]),
            'overfitting_detected': overfitting_detected,
            'overfitting_epoch': overfitting_epoch,
            'val_train_gap': float(val_arr[-1] - train_arr[-1]),
            'val_train_ratio': float(val_arr[-1] / (train_arr[-1] + 1e-8))
        }
        
        return analysis
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_metrics: Optional[Dict[str, List[float]]] = None,
        val_metrics: Optional[Dict[str, List[float]]] = None,
        epochs: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> Figure:
        """
        Plot training curves
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
            epochs: Corresponding epoch numbers
            save_path: Save path
            show: Whether to display
            
        Returns:
            Figure object
        """
        if epochs is None:
            epochs = list(range(1, len(train_losses) + 1))
        
        # Create figure
        num_plots = 1 + (1 if train_metrics else 0)
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots))
        
        if num_plots == 1:
            axes = [axes]
        
        # Plot loss curves
        ax = axes[0]
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mark best validation point
        min_val_idx = np.argmin(val_losses)
        ax.scatter(epochs[min_val_idx], val_losses[min_val_idx], 
                  color='red', s=100, zorder=5, label=f'Best Val Loss (epoch {epochs[min_val_idx]})')
        ax.legend()
        
        # Plot metric curves (if available)
        if train_metrics and val_metrics and num_plots > 1:
            ax = axes[1]
            for metric_name in train_metrics.keys():
                if metric_name in val_metrics:
                    ax.plot(epochs, train_metrics[metric_name], '--', 
                           label=f'Training {metric_name}', alpha=0.7)
                    ax.plot(epochs, val_metrics[metric_name], '-', 
                           label=f'Validation {metric_name}', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Metric Value')
            ax.set_title('Training and Validation Metric Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def analyze_learning_rate(
        self,
        learning_rates: List[float],
        losses: List[float],
        epochs: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Analyze learning rate impact on loss
        
        Args:
            learning_rates: List of learning rates
            losses: Corresponding losses
            epochs: Epoch numbers
            
        Returns:
            Learning rate analysis results
        """
        if epochs is None:
            epochs = list(range(1, len(learning_rates) + 1))
        
        if len(learning_rates) != len(losses):
            raise ValueError("Learning rates and losses must have same length")
        
        # Convert to numpy arrays
        lr_arr = np.array(learning_rates)
        loss_arr = np.array(losses)
        
        # Calculate learning rate changes
        lr_changes = np.diff(lr_arr) / lr_arr[:-1]
        
        # Calculate loss changes
        loss_changes = np.diff(loss_arr) / (loss_arr[:-1] + 1e-8)
        
        # Analyze relationship between learning rate and loss
        analysis = {
            'epochs': epochs,
            'learning_rates': learning_rates,
            'losses': losses,
            'lr_changes': lr_changes.tolist(),
            'loss_changes': loss_changes.tolist(),
            'final_lr': float(lr_arr[-1]),
            'final_loss': float(loss_arr[-1]),
            'lr_range': [float(lr_arr.min()), float(lr_arr.max())],
            'suggested_lr': self._suggest_learning_rate(lr_arr, loss_arr)
        }
        
        return analysis
    
    def _suggest_learning_rate(self, lr_arr: np.ndarray, loss_arr: np.ndarray) -> float:
        """Suggest learning rate"""
        if len(lr_arr) < 5:
            return float(lr_arr[-1])
        
        # Simple heuristic: find learning rate range with fastest loss decrease
        window_size = min(5, len(lr_arr) // 2)
        best_lr = lr_arr[-1]
        best_improvement = 0
        
        for i in range(len(lr_arr) - window_size):
            window_lr = lr_arr[i:i+window_size]
            window_loss = loss_arr[i:i+window_size]
            
            # Calculate loss improvement
            loss_improvement = window_loss[0] - window_loss[-1]
            avg_lr = np.mean(window_lr)
            
            if loss_improvement > best_improvement:
                best_improvement = loss_improvement
                best_lr = avg_lr
        
        return float(best_lr)
    
    def generate_training_report(
        self,
        train_history: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate training report
        
        Args:
            train_history: Training history dictionary
            save_path: Save path
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("Training Analysis Report")
        report_lines.append("=" * 60)
        
        # Extract basic information
        epochs = train_history.get('epochs', [])
        train_losses = train_history.get('train_losses', [])
        val_losses = train_history.get('val_losses', [])
        
        if epochs and train_losses and val_losses:
            report_lines.append(f"\nTraining Epochs: {len(epochs)}")
            report_lines.append(f"Final Training Loss: {train_losses[-1]:.6f}")
            report_lines.append(f"Final Validation Loss: {val_losses[-1]:.6f}")
            
            # Overfitting analysis
            overfitting_analysis = self.analyze_overfitting(train_losses, val_losses, epochs)
            if overfitting_analysis['overfitting_detected']:
                report_lines.append(f"Warning: Overfitting detected (epoch {overfitting_analysis['overfitting_epoch']})")
            else:
                report_lines.append("No significant overfitting detected")
            
            # Best model
            report_lines.append(f"Best Validation Loss: {overfitting_analysis['min_val_loss']:.6f} (epoch {overfitting_analysis['min_val_epoch']})")
        
        # Learning rate analysis (if available)
        if 'learning_rates' in train_history:
            lr_analysis = self.analyze_learning_rate(
                train_history['learning_rates'],
                train_losses if train_losses else [0] * len(train_history['learning_rates']),
                epochs
            )
            report_lines.append(f"\nLearning Rate Analysis:")
            report_lines.append(f"  Final Learning Rate: {lr_analysis['final_lr']:.2e}")
            report_lines.append(f"  Suggested Learning Rate: {lr_analysis['suggested_lr']:.2e}")
        
        report_lines.append("\n" + "=" * 60)
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Training report saved to: {save_path}")
        
        return report_text


__all__ = ['TrainingCurveAnalyzer']
