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
from scipy import stats

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
    
    def analyze_weights(
        self,
        model: nn.Module,
        previous_weights: Optional[Dict[str, torch.Tensor]] = None,
        detect_anomalies: bool = True,
        anomaly_threshold: float = 3.0
    ) -> Dict[str, Any]:
        """
        分析权重分布和变化，检测权重异常
        
        参数:
            model: 神经网络模型
            previous_weights: 之前的权重字典（用于跟踪变化），如果为None则不跟踪变化
            detect_anomalies: 是否检测权重异常（过大/过小的权重）
            anomaly_threshold: 异常值检测的标准差倍数阈值
            
        返回:
            权重分析报告
        """
        if not self.config.check_weights:
            return {"enabled": False}
        
        weight_stats = {}
        all_weights = []
        weight_changes = {}
        
        current_weights = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                weights = param.data.cpu().numpy().flatten()
                all_weights.extend(weights)
                current_weights[name] = param.data.clone()
                
                # 基本统计
                weight_mean = np.mean(weights)
                weight_std = np.std(weights)
                weight_min = np.min(weights)
                weight_max = np.max(weights)
                weight_abs_mean = np.mean(np.abs(weights))
                weight_median = np.median(weights)
                
                # 检测异常值
                anomalies_info = {}
                if detect_anomalies and len(weights) > 10:
                    if weight_std > 1e-10:  # 避免除零
                        z_scores = np.abs((weights - weight_mean) / weight_std)
                        anomaly_mask = z_scores > anomaly_threshold
                        anomaly_count = np.sum(anomaly_mask)
                        anomaly_ratio = anomaly_count / len(weights)
                        
                        # 检测过大/过小的权重
                        large_weights = np.sum(weights > weight_mean + anomaly_threshold * weight_std)
                        small_weights = np.sum(weights < weight_mean - anomaly_threshold * weight_std)
                        
                        anomalies_info = {
                            'anomaly_count': int(anomaly_count),
                            'anomaly_ratio': float(anomaly_ratio),
                            'large_weight_count': int(large_weights),
                            'small_weight_count': int(small_weights),
                            'max_z_score': float(np.max(z_scores)),
                            'anomaly_threshold': anomaly_threshold
                        }
                
                # 计算权重变化（如果提供了之前的权重）
                change_info = {}
                if previous_weights is not None and name in previous_weights:
                    prev_weight = previous_weights[name].cpu().numpy().flatten()
                    curr_weight = weights
                    
                    if len(prev_weight) == len(curr_weight):
                        # 计算绝对变化和相对变化
                        abs_change = np.abs(curr_weight - prev_weight)
                        rel_change = abs_change / (np.abs(prev_weight) + 1e-10)
                        
                        change_info = {
                            'mean_abs_change': float(np.mean(abs_change)),
                            'max_abs_change': float(np.max(abs_change)),
                            'mean_rel_change': float(np.mean(rel_change)),
                            'max_rel_change': float(np.max(rel_change)),
                            'change_std': float(np.std(abs_change))
                        }
                        
                        weight_changes[name] = change_info
                
                weight_stats[name] = {
                    'mean': float(weight_mean),
                    'std': float(weight_std),
                    'min': float(weight_min),
                    'max': float(weight_max),
                    'abs_mean': float(weight_abs_mean),
                    'median': float(weight_median),
                    'shape': list(param.shape),
                    'num_params': len(weights),
                    **anomalies_info,
                    **change_info
                }
        
        if not all_weights:
            return {"error": "未找到可分析的权重"}
        
        all_weights = np.array(all_weights)
        
        # 全局统计
        global_mean = np.mean(all_weights)
        global_std = np.std(all_weights)
        global_min = np.min(all_weights)
        global_max = np.max(all_weights)
        
        # 检测全局权重问题
        weight_issues = []
        
        # 检测权重初始化问题
        if abs(global_mean) > 0.1:
            weight_issues.append(f"权重均值较大 ({global_mean:.4f})，可能初始化不当")
        
        if global_std > 1.0:
            weight_issues.append(f"权重标准差较大 ({global_std:.4f})，可能导致梯度爆炸")
        
        if global_std < 1e-4:
            weight_issues.append(f"权重标准差过小 ({global_std:.4f})，可能导致梯度消失")
        
        # 检测权重范围问题
        weight_range = global_max - global_min
        if weight_range > 10.0:
            weight_issues.append(f"权重范围过大 ({weight_range:.4f})")
        
        # 检测权重分布问题
        if len(all_weights) > 1:
            skewness = stats.skew(all_weights)
            kurtosis = stats.kurtosis(all_weights)
            
            if abs(skewness) > 2.0:
                weight_issues.append(f"权重分布严重偏斜 (偏度: {skewness:.2f})")
            
            if kurtosis > 5.0:
                weight_issues.append(f"权重分布尖峰 (峰度: {kurtosis:.2f})")
        
        # 计算权重变化趋势（如果有之前的权重）
        change_trend = {}
        if weight_changes:
            all_abs_changes = [info['mean_abs_change'] for info in weight_changes.values()]
            all_rel_changes = [info['mean_rel_change'] for info in weight_changes.values()]
            
            change_trend = {
                'mean_abs_change': float(np.mean(all_abs_changes)) if all_abs_changes else 0.0,
                'max_abs_change': float(np.max(all_abs_changes)) if all_abs_changes else 0.0,
                'mean_rel_change': float(np.mean(all_rel_changes)) if all_rel_changes else 0.0,
                'max_rel_change': float(np.max(all_rel_changes)) if all_rel_changes else 0.0,
                'num_layers_with_changes': len(weight_changes)
            }
        
        return {
            'global_stats': {
                'mean': float(global_mean),
                'std': float(global_std),
                'min': float(global_min),
                'max': float(global_max),
                'abs_mean': float(np.mean(np.abs(all_weights))),
                'median': float(np.median(all_weights)),
                'q25': float(np.percentile(all_weights, 25)),
                'q75': float(np.percentile(all_weights, 75)),
                'skewness': float(stats.skew(all_weights) if len(all_weights) > 1 else 0),
                'kurtosis': float(stats.kurtosis(all_weights) if len(all_weights) > 1 else 0),
                'weight_range': float(weight_range),
                'total_params': len(all_weights)
            },
            'per_layer_stats': weight_stats,
            'weight_changes': weight_changes,
            'change_trend': change_trend,
            'weight_issues': weight_issues,
            'current_weights': {k: v.shape for k, v in current_weights.items()} if current_weights else {},
            'num_layers': len(weight_stats)
        }
    
    def analyze_gradients(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        compute_norms: bool = True,
        detect_outliers: bool = True,
        outlier_threshold: float = 3.0
    ) -> Dict[str, Any]:
        """
        分析梯度问题，包括梯度消失/爆炸、梯度范数、统计信息和异常值检测
        
        参数:
            model: 神经网络模型
            loss: 损失张量
            compute_norms: 是否计算梯度范数（L1, L2, 无穷范数）
            detect_outliers: 是否检测梯度异常值
            outlier_threshold: 异常值检测的标准差倍数阈值
            
        返回:
            梯度诊断报告
        """
        if not self.config.check_gradients:
            return {"enabled": False}
        
        # 计算梯度
        loss.backward(retain_graph=True)
        
        gradient_stats = {}
        all_gradients = []
        layer_gradients = {}
        
        # 初始化统计信息
        total_l1_norm = 0.0
        total_l2_norm = 0.0
        total_inf_norm = 0.0
        max_grad = -float('inf')
        min_grad = float('inf')
        zero_grad_count = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                grad_flat = grad.flatten()
                
                # 收集所有梯度值用于全局统计
                all_gradients.extend(grad_flat.cpu().numpy())
                layer_gradients[name] = grad_flat.cpu().numpy()
                
                # 计算梯度范数
                l1_norm = grad.abs().sum().item()
                l2_norm = grad.norm().item()
                inf_norm = grad.abs().max().item()
                
                total_l1_norm += l1_norm
                total_l2_norm += l2_norm ** 2  # 平方和，最后开方
                total_inf_norm = max(total_inf_norm, inf_norm)
                
                # 基本统计
                grad_max = grad.max().item()
                grad_min = grad.min().item()
                grad_mean = grad.mean().item()
                grad_std = grad.std(unbiased=False).item()
                grad_median = grad.median().item()
                
                max_grad = max(max_grad, grad_max)
                min_grad = min(min_grad, grad_min)
                
                # 统计接近零的梯度
                zero_mask = torch.abs(grad) < 1e-6
                zero_count = zero_mask.sum().item()
                zero_grad_count += zero_count
                total_params += grad.numel()
                
                # 检测异常值（如果启用）
                outliers_info = {}
                if detect_outliers and len(grad_flat) > 10:
                    grad_np = grad_flat.cpu().numpy()
                    grad_mean_np = np.mean(grad_np)
                    grad_std_np = np.std(grad_np)
                    
                    if grad_std_np > 1e-10:  # 避免除零
                        z_scores = np.abs((grad_np - grad_mean_np) / grad_std_np)
                        outlier_mask = z_scores > outlier_threshold
                        outlier_count = np.sum(outlier_mask)
                        outlier_ratio = outlier_count / len(grad_np)
                        
                        outliers_info = {
                            'outlier_count': int(outlier_count),
                            'outlier_ratio': float(outlier_ratio),
                            'max_z_score': float(np.max(z_scores)),
                            'outlier_threshold': outlier_threshold
                        }
                
                gradient_stats[name] = {
                    'l1_norm': l1_norm,
                    'l2_norm': l2_norm,
                    'inf_norm': inf_norm,
                    'max': grad_max,
                    'min': grad_min,
                    'mean': grad_mean,
                    'std': grad_std,
                    'median': grad_median,
                    'zero_ratio': zero_count / grad.numel() if grad.numel() > 0 else 0,
                    'shape': list(grad.shape),
                    'num_params': grad.numel(),
                    **outliers_info
                }
        
        # 计算总L2范数（平方和开方）
        total_l2_norm = total_l2_norm ** 0.5
        
        # 全局统计
        all_gradients = np.array(all_gradients)
        global_stats = {}
        if len(all_gradients) > 0:
            global_stats = {
                'mean': float(np.mean(all_gradients)),
                'std': float(np.std(all_gradients)),
                'min': float(np.min(all_gradients)),
                'max': float(np.max(all_gradients)),
                'median': float(np.median(all_gradients)),
                'q25': float(np.percentile(all_gradients, 25)),
                'q75': float(np.percentile(all_gradients, 75)),
                'skewness': float(stats.skew(all_gradients) if len(all_gradients) > 1 else 0),
                'kurtosis': float(stats.kurtosis(all_gradients) if len(all_gradients) > 1 else 0)
            }
        
        # 检测梯度消失/爆炸
        gradient_vanishing = total_l2_norm < 1e-6
        gradient_exploding = total_l2_norm > 1e3
        
        # 检测梯度分布问题
        gradient_issues = []
        if gradient_vanishing:
            gradient_issues.append("梯度消失：总梯度范数过小")
        if gradient_exploding:
            gradient_issues.append("梯度爆炸：总梯度范数过大")
        
        if len(all_gradients) > 0:
            zero_ratio = np.sum(np.abs(all_gradients) < 1e-6) / len(all_gradients)
            if zero_ratio > 0.9:
                gradient_issues.append(f"梯度稀疏：{zero_ratio:.1%}的梯度接近零")
            
            # 检查梯度分布是否对称
            if global_stats.get('skewness', 0) > 2.0:
                gradient_issues.append(f"梯度分布严重右偏（偏度：{global_stats['skewness']:.2f}）")
            elif global_stats.get('skewness', 0) < -2.0:
                gradient_issues.append(f"梯度分布严重左偏（偏度：{global_stats['skewness']:.2f}）")
        
        # 清除梯度以避免影响后续训练
        model.zero_grad()
        
        return {
            'total_l1_norm': total_l1_norm,
            'total_l2_norm': total_l2_norm,
            'total_inf_norm': total_inf_norm,
            'max_gradient': max_grad,
            'min_gradient': min_grad,
            'zero_gradient_ratio': zero_grad_count / total_params if total_params > 0 else 0,
            'gradient_vanishing': gradient_vanishing,
            'gradient_exploding': gradient_exploding,
            'gradient_issues': gradient_issues,
            'global_stats': global_stats,
            'per_layer_stats': gradient_stats,
            'total_params': total_params,
            'num_layers': len(gradient_stats)
        }
    
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
                'std_activation': float(activation_flat.std(unbiased=False).item())
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
    
    def detect_dead_relu_neurons(
        self,
        model: nn.Module,
        sample_inputs: List[torch.Tensor],
        threshold: float = 1e-6
    ) -> Dict[str, Any]:
        """
        检测死ReLU神经元（支持多个样本输入）
        
        参数:
            model: PyTorch模型
            sample_inputs: 样本输入列表（多个批次）
            threshold: 死神经元阈值
            
        返回:
            ReLU分析字典，包含overall_dead_ratio字段
        """
        # 如果只有一个输入，直接使用check_dead_relu
        if len(sample_inputs) == 1:
            report = self.check_dead_relu(model, sample_inputs[0], threshold)
            # 确保包含overall_dead_ratio字段
            report['overall_dead_ratio'] = report.get('global_dead_ratio', 0.0)
            return report
        
        # 多个输入：拼接成一个大的批次
        # 检查所有输入是否具有相同的形状（除了批次维度）
        try:
            # 尝试在批次维度上拼接
            combined_input = torch.cat(sample_inputs, dim=0)
        except Exception as e:
            # 如果拼接失败，则对每个输入单独处理并聚合结果
            print(f"警告: 无法拼接输入，将单独处理每个样本: {e}")
            return self._detect_dead_relu_multiple_inputs(model, sample_inputs, threshold)
        
        # 使用拼接后的输入运行check_dead_relu
        report = self.check_dead_relu(model, combined_input, threshold)
        report['overall_dead_ratio'] = report.get('global_dead_ratio', 0.0)
        return report
    
    def _detect_dead_relu_multiple_inputs(
        self,
        model: nn.Module,
        sample_inputs: List[torch.Tensor],
        threshold: float = 1e-6
    ) -> Dict[str, Any]:
        """
        处理多个样本输入的死ReLU检测（当无法拼接时）
        
        对每个输入单独运行检测，然后聚合结果
        """
        all_reports = []
        for i, inp in enumerate(sample_inputs):
            report = self.check_dead_relu(model, inp, threshold)
            all_reports.append(report)
        
        # 聚合结果：计算平均死神经元比例
        if not all_reports:
            return {
                'overall_dead_ratio': 0.0,
                'global_dead_ratio': 0.0,
                'total_dead_neurons': 0,
                'total_neurons': 0,
                'per_layer_stats': {},
                'has_dead_relu': False
            }
        
        # 合并每层统计（取平均值）
        per_layer_stats = {}
        total_dead_neurons = 0
        total_neurons = 0
        
        # 收集所有层名
        layer_names = set()
        for report in all_reports:
            if 'per_layer_stats' in report:
                layer_names.update(report['per_layer_stats'].keys())
        
        for layer_name in layer_names:
            dead_neurons_list = []
            total_neurons_list = []
            dead_ratio_list = []
            
            for report in all_reports:
                if 'per_layer_stats' in report and layer_name in report['per_layer_stats']:
                    stats = report['per_layer_stats'][layer_name]
                    dead_neurons_list.append(stats['dead_neurons'])
                    total_neurons_list.append(stats['total_neurons'])
                    dead_ratio_list.append(stats['dead_ratio'])
            
            if dead_neurons_list:
                avg_dead_neurons = sum(dead_neurons_list) / len(dead_neurons_list)
                avg_total_neurons = sum(total_neurons_list) / len(total_neurons_list)
                avg_dead_ratio = sum(dead_ratio_list) / len(dead_ratio_list)
                
                per_layer_stats[layer_name] = {
                    'dead_neurons': int(avg_dead_neurons),
                    'total_neurons': int(avg_total_neurons),
                    'dead_ratio': avg_dead_ratio,
                    'mean_activation': 0.0,  # 无法聚合
                    'std_activation': 0.0
                }
                
                total_dead_neurons += int(avg_dead_neurons)
                total_neurons += int(avg_total_neurons)
        
        overall_dead_ratio = total_dead_neurons / total_neurons if total_neurons > 0 else 0.0
        
        return {
            'per_layer_stats': per_layer_stats,
            'total_dead_neurons': total_dead_neurons,
            'total_neurons': total_neurons,
            'global_dead_ratio': overall_dead_ratio,
            'overall_dead_ratio': overall_dead_ratio,
            'has_dead_relu': total_dead_neurons > 0
        }
    
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
