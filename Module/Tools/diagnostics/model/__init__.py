"""
模型诊断模块

提供ModelDiagnostics类，用于分析模型权重、梯度和激活值。
"""

import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DiagnosticsConfig


class ModelDiagnostics:
    """
    模型诊断工具
    
    分析以下方面：
    - 权重分布和统计
    - 梯度问题（消失/爆炸）
    - 激活值分布
    - 死ReLU检测
    """
    
    def __init__(self, config: Optional[DiagnosticsConfig] = None):
        """
        初始化模型诊断工具
        
        参数:
            config: 诊断配置，如果为None则使用默认配置
        """
        self.config = config or DiagnosticsConfig()
    
    def analyze_weight_distribution(
        self,
        model: nn.Module,
        layer_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        分析模型权重分布
        
        参数:
            model: PyTorch模型
            layer_types: 要分析的层类型列表，如果为None则分析所有层
            
        返回:
            权重统计字典
        """
        if layer_types is None:
            layer_types = ['Linear', 'Conv2d', 'Conv3d', 'BatchNorm']
        
        stats = {
            'total_params': 0,
            'trainable_params': 0,
            'per_layer_stats': {},
            'global_stats': {}
        }
        
        all_weights = []
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                # 检查层类型
                layer_name = name.split('.')[-2] if '.' in name else name
                layer_type = self._get_layer_type(model, name)
                
                if layer_type in layer_types or not layer_types:
                    # 收集权重
                    weights = param.data.cpu().numpy().flatten()
                    all_weights.extend(weights)
                    
                    # 计算层统计
                    layer_stats = {
                        'mean': float(np.mean(weights)),
                        'std': float(np.std(weights)),
                        'min': float(np.min(weights)),
                        'max': float(np.max(weights)),
                        'abs_mean': float(np.mean(np.abs(weights))),
                        'num_params': len(weights),
                        'layer_type': layer_type
                    }
                    
                    stats['per_layer_stats'][name] = layer_stats
                    stats['total_params'] += len(weights)
                    if param.requires_grad:
                        stats['trainable_params'] += len(weights)
        
        # 计算全局统计
        if all_weights:
            all_weights = np.array(all_weights)
            stats['global_stats'] = {
                'mean': float(np.mean(all_weights)),
                'std': float(np.std(all_weights)),
                'min': float(np.min(all_weights)),
                'max': float(np.max(all_weights)),
                'abs_mean': float(np.mean(np.abs(all_weights))),
                'percentile_5': float(np.percentile(all_weights, 5)),
                'percentile_95': float(np.percentile(all_weights, 95))
            }
        
        return stats
    
    def check_gradient_issues(
        self,
        model: nn.Module,
        loss: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        检查梯度问题（消失/爆炸梯度）
        
        参数:
            model: PyTorch模型
            loss: 损失值，如果提供则计算梯度
            
        返回:
            梯度分析字典
        """
        gradient_stats = {
            'has_gradients': False,
            'per_layer_grads': {},
            'global_grad_stats': {},
            'issues': []
        }
        
        # 如果有损失，计算梯度
        if loss is not None:
            # 清零梯度
            model.zero_grad()
            
            # 计算梯度
            loss.backward()
        
        all_gradients = []
        
        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                gradient_stats['has_gradients'] = True
                
                # 获取梯度
                grads = param.grad.data.cpu().numpy().flatten()
                all_gradients.extend(grads)
                
                # 计算梯度范数
                grad_norm = np.linalg.norm(grads)
                grad_mean = np.mean(grads)
                grad_std = np.std(grads)
                grad_abs_mean = np.mean(np.abs(grads))
                
                # 检查梯度问题
                issues = []
                if grad_abs_mean < 1e-7:
                    issues.append('vanishing_gradient')
                if grad_abs_mean > 100.0:
                    issues.append('exploding_gradient')
                if np.any(np.isnan(grads)):
                    issues.append('nan_gradients')
                if np.any(np.isinf(grads)):
                    issues.append('inf_gradients')
                
                layer_stats = {
                    'norm': float(grad_norm),
                    'mean': float(grad_mean),
                    'std': float(grad_std),
                    'abs_mean': float(grad_abs_mean),
                    'min': float(np.min(grads)),
                    'max': float(np.max(grads)),
                    'issues': issues
                }
                
                gradient_stats['per_layer_grads'][name] = layer_stats
                
                if issues:
                    gradient_stats['issues'].extend([f"{name}: {issue}" for issue in issues])
        
        # 计算全局梯度统计
        if all_gradients:
            all_gradients = np.array(all_gradients)
            gradient_stats['global_grad_stats'] = {
                'total_norm': float(np.linalg.norm(all_gradients)),
                'mean': float(np.mean(all_gradients)),
                'std': float(np.std(all_gradients)),
                'abs_mean': float(np.mean(np.abs(all_gradients))),
                'min': float(np.min(all_gradients)),
                'max': float(np.max(all_gradients))
            }
        
        return gradient_stats
    
    def check_dead_relu(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        threshold: float = 1e-7
    ) -> Dict[str, Any]:
        """
        检查死ReLU问题
        
        参数:
            model: PyTorch模型
            sample_input: 样本输入
            threshold: 死神经元阈值
            
        返回:
            ReLU分析字典
        """
        # 注册钩子来捕获激活值
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach()
            return hook
        
        # 注册ReLU层的钩子
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_forward_hook(get_activation(name))
                hooks.append(hook)
        
        # 前向传播
        with torch.no_grad():
            _ = model(sample_input)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 分析激活值
        relu_stats = {}
        total_dead_neurons = 0
        total_neurons = 0
        
        for name, activation in activations.items():
            # 展激活值
            if activation.dim() > 2:
                activation_flat = activation.view(activation.size(0), -1)
            else:
                activation_flat = activation
            
            # 计算死神经元比例
            dead_mask = (activation_flat.abs() < threshold).all(dim=0)
            dead_count = dead_mask.sum().item()
            total_neurons_layer = activation_flat.size(1)
            
            dead_ratio = dead_count / total_neurons_layer if total_neurons_layer > 0 else 0.0
            
            relu_stats[name] = {
                'dead_neurons': int(dead_count),
                'total_neurons': total_neurons_layer,
                'dead_ratio': float(dead_ratio),
                'mean_activation': float(activation_flat.mean().item()),
                'std_activation': float(activation_flat.std().item())
            }
            
            total_dead_neurons += dead_count
            total_neurons += total_neurons_layer
        
        # 汇总统计
        summary = {
            'per_layer_stats': relu_stats,
            'total_dead_neurons': total_dead_neurons,
            'total_neurons': total_neurons,
            'global_dead_ratio': total_dead_neurons / total_neurons if total_neurons > 0 else 0.0,
            'has_dead_relu': total_dead_neurons > 0
        }
        
        return summary
    
    def _get_layer_type(self, model: nn.Module, param_name: str) -> str:
        """获取参数对应的层类型"""
        # 简化实现：从参数名推断层类型
        if 'conv' in param_name.lower():
            return 'Conv'
        elif 'linear' in param_name.lower() or 'fc' in param_name.lower():
            return 'Linear'
        elif 'norm' in param_name.lower() or 'bn' in param_name.lower():
            return 'BatchNorm'
        elif 'weight' in param_name:
            # 尝试从模块名获取
            parts = param_name.split('.')
            if len(parts) >= 2:
                module_name = parts[-2]
                for name, module in model.named_modules():
                    if name == module_name:
                        return module.__class__.__name__
        return 'Unknown'


__all__ = ['ModelDiagnostics']
