"""
训练脚本
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
import yaml
from datetime import datetime
from typing import Optional

from ..Config.config import Config
from ..Loader.data_loader import prepare_data_loaders, create_dummy_data
from .models import create_model
from ..Tools.utils import save_checkpoint, load_checkpoint, calculate_metrics
from ..Tools.diagnostics import (
    DiagnosticsConfig, ImageMetricsCalculator, ValidationVisualizer,
    ModelDiagnostics, TrainingCurveAnalyzer
)
from ..Tools.amp_optimizer import AMPOptimizer, autocast_context, get_recommended_amp_config


class Trainer:
    """训练器类"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        
        # 创建目录
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        os.makedirs(config.training.log_dir, exist_ok=True)
        
        # 初始化模型、优化器、损失函数
        self.model = create_model(config.model).to(self.device)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_loss_function()
        
        # 数据加载器
        self.train_loader, self.val_loader, self.test_loader = prepare_data_loaders(config.data)
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.writer = SummaryWriter(log_dir=config.training.log_dir)
        
        # 早停机制
        self.early_stopping_counter = 0
        self.early_stopping_patience = config.training.early_stopping_patience if hasattr(config.training, 'early_stopping_patience') else config.training.patience
        self.use_early_stopping = getattr(config.training, 'use_early_stopping', True)
        
        # 梯度裁剪
        self.gradient_clip_value = getattr(config.training, 'gradient_clip_value', None)
        self.gradient_clip_norm = getattr(config.training, 'gradient_clip_norm', None)
        
        # 学习率热身
        self.warmup_epochs = getattr(config.training, 'warmup_epochs', 0)
        self.warmup_scheduler = None
        if self.warmup_epochs > 0:
            self.warmup_scheduler = self._create_warmup_scheduler()
        
        # 混合精度训练
        self.use_amp = getattr(config.training, 'use_amp', False)
        self.scaler = None
        if self.use_amp:
            self.scaler = torch.amp.GradScaler()
        
        # 日志和监控
        self.log_interval = getattr(config.training, 'log_interval', 10)
        self.save_interval = getattr(config.training, 'save_interval', 10)
        
        # 训练监控
        self.train_losses = []
        self.val_losses = []
        self.val_psnrs = []
        self.val_ssims = []
        self.learning_rates = []
        
        # 诊断工具初始化
        self.enable_diagnostics = getattr(config.diagnostics, 'enable_diagnostics', True)
        if self.enable_diagnostics:
            print("初始化诊断工具...")
            # 创建诊断配置
            self.diagnostics_config = DiagnosticsConfig(
                enable_diagnostics=config.diagnostics.enable_diagnostics,
                compute_rmse=getattr(config.diagnostics, 'compute_rmse', True),
                compute_mae=getattr(config.diagnostics, 'compute_mae', True),
                compute_psnr=getattr(config.diagnostics, 'compute_psnr', True),
                compute_ssim=getattr(config.diagnostics, 'compute_ssim', True),
                compute_ms_ssim=getattr(config.diagnostics, 'compute_ms_ssim', False),
                compute_lpips=getattr(config.diagnostics, 'compute_lpips', False),
                visualize_samples=getattr(config.diagnostics, 'visualize_samples', 5),
                save_visualizations=getattr(config.diagnostics, 'save_visualizations', True),
                visualization_dir=getattr(config.diagnostics, 'visualization_dir', "./diagnostics/visualizations"),
                dpi=getattr(config.diagnostics, 'dpi', 150),
                visualization_frequency=getattr(config.diagnostics, 'visualization_frequency', 5),
                check_gradients=getattr(config.diagnostics, 'check_gradients', True),
                check_weights=getattr(config.diagnostics, 'check_weights', True),
                check_activations=getattr(config.diagnostics, 'check_activations', False),
                check_dead_relu=getattr(config.diagnostics, 'check_dead_relu', True),
                model_diagnosis_frequency=getattr(config.diagnostics, 'model_diagnosis_frequency', 10),
                analyze_overfitting=getattr(config.diagnostics, 'analyze_overfitting', True),
                compute_loss_ratio=getattr(config.diagnostics, 'compute_loss_ratio', True),
                check_learning_rate=getattr(config.diagnostics, 'check_learning_rate', True),
                training_analysis_frequency=getattr(config.diagnostics, 'training_analysis_frequency', 5),
                generate_html_report=getattr(config.diagnostics, 'generate_html_report', True),
                generate_pdf_report=getattr(config.diagnostics, 'generate_pdf_report', False),
                report_dir=getattr(config.diagnostics, 'report_dir', "./diagnostics/reports")
            )
            
            # 初始化诊断工具
            self.metrics_calculator = ImageMetricsCalculator(self.diagnostics_config)
            self.visualizer = ValidationVisualizer(self.diagnostics_config)
            self.model_diagnostics = ModelDiagnostics(self.diagnostics_config)
            self.training_analyzer = TrainingCurveAnalyzer(self.diagnostics_config)
            
            # 诊断状态
            self.detailed_metrics_history = []
            self.model_diagnosis_history = []
            self.training_analysis_history = []
            
            print(f"诊断工具已初始化: 可视化频率={self.diagnostics_config.visualization_frequency} epoch, "
                  f"模型诊断频率={self.diagnostics_config.model_diagnosis_frequency} epoch")
        else:
            self.diagnostics_config = None
            self.metrics_calculator = None
            self.visualizer = None
            self.model_diagnostics = None
            self.training_analyzer = None
            print("诊断功能已禁用")
        
        print(f"设备: {self.device}")
        print(f"模型: {config.model.model_name}")
        print(f"参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"优化器: {config.training.optimizer}")
        print(f"损失函数: {config.training.loss_function}")
        if self.config.training.use_multi_scale_loss:
            print(f"使用多尺度损失，权重: {self.config.training.multi_scale_weights}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        if self.config.training.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"未知优化器: {self.config.training.optimizer}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        scheduler_name = self.config.training.scheduler.lower()
        
        # 获取调度器参数
        factor = getattr(self.config.training, 'scheduler_factor', 0.5)
        step_size = getattr(self.config.training, 'scheduler_step_size', 20)
        gamma = getattr(self.config.training, 'scheduler_gamma', 0.5)
        milestones = getattr(self.config.training, 'scheduler_milestones', (30, 60, 90))
        
        if scheduler_name == "reducelronplateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=factor,
                patience=self.config.training.patience // 2,
                min_lr=self.config.training.min_lr
            )
        elif scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs - self.warmup_epochs,
                eta_min=self.config.training.min_lr
            )
        elif scheduler_name == "cosinewarmrestarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=self.config.training.min_lr
            )
        elif scheduler_name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_name == "multistep":
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=gamma
            )
        else:
            return None
    
    def _create_warmup_scheduler(self):
        """创建学习率热身调度器"""
        from torch.optim.lr_scheduler import LambdaLR
        
        def warmup_lambda(epoch):
            if epoch < self.warmup_epochs:
                return float(epoch + 1) / float(self.warmup_epochs)
            return 1.0
        
        return LambdaLR(self.optimizer, lr_lambda=warmup_lambda)
    
    def _apply_gradient_clipping(self):
        """应用梯度裁剪"""
        if self.gradient_clip_value is not None:
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(),
                self.gradient_clip_value
            )
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm
            )
    
    def _check_early_stopping(self, val_loss):
        """检查是否应该早停"""
        if not self.use_early_stopping:
            return False
            
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"早停触发！验证损失在 {self.early_stopping_patience} 个epoch内未改善")
                return True
            return False
    
    def _log_gradient_norms(self, epoch):
        """记录梯度范数"""
        total_norm = 0.0
        max_norm = 0.0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_norm = max(max_norm, param_norm)
                
                # 记录每个参数的梯度范数
                self.writer.add_scalar(
                    f"gradients/{name}_norm",
                    param_norm,
                    epoch
                )
        
        total_norm = total_norm ** 0.5
        self.writer.add_scalar("gradients/total_norm", total_norm, epoch)
        self.writer.add_scalar("gradients/max_norm", max_norm, epoch)
        
        return total_norm, max_norm
    
    def _log_weight_histograms(self, epoch):
        """记录权重直方图"""
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(f"weights/{name}", param.data, epoch)
            if param.grad is not None:
                self.writer.add_histogram(f"gradients/{name}", param.grad.data, epoch)
    
    def _create_loss_function(self):
        """创建损失函数"""
        from .losses import create_loss_function
        
        loss_name = self.config.training.loss_function
        kwargs = {}
        
        # 根据损失函数类型设置参数
        if loss_name.lower() == "mixedloss":
            kwargs['weights'] = self.config.training.loss_weights
        elif loss_name.lower() == "multiscaleloss":
            kwargs['base_loss'] = "L1Loss"
            kwargs['scales'] = (1, 2, 4)
            kwargs['weights'] = self.config.training.multi_scale_weights
        elif loss_name.lower() == "ssimloss":
            kwargs['window_size'] = 11
            kwargs['sigma'] = 1.5
            kwargs['data_range'] = 1.0
        
        # 如果启用了多尺度损失，使用MultiScaleLoss包装器
        if self.config.training.use_multi_scale_loss:
            base_loss_name = loss_name
            if base_loss_name.lower() == "mixedloss":
                kwargs['weights'] = self.config.training.loss_weights
                base_loss = create_loss_function("MixedLoss", **kwargs)
            else:
                base_loss = create_loss_function(base_loss_name, **kwargs)
            
            return create_loss_function(
                "MultiScaleLoss",
                base_loss=base_loss_name,
                scales=(1, 2, 4),
                weights=self.config.training.multi_scale_weights
            )
        
        return create_loss_function(loss_name, **kwargs)
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"训练 Epoch {self.current_epoch}")
        
        for batch_idx, (low_dose, full_dose) in enumerate(progress_bar):
            low_dose = low_dose.to(self.device)
            full_dose = full_dose.to(self.device)
            
            # 混合精度训练
            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    enhanced = self.model(low_dose)
                    loss = self.criterion(enhanced, full_dose)
                
                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # 应用梯度裁剪
                self.scaler.unscale_(self.optimizer)
                self._apply_gradient_clipping()
                
                # 优化器更新
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 标准训练
                self.optimizer.zero_grad()
                enhanced = self.model(low_dose)
                loss = self.criterion(enhanced, full_dose)
                
                # 反向传播
                loss.backward()
                
                # 应用梯度裁剪
                self._apply_gradient_clipping()
                
                # 优化器更新
                self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            # TensorBoard记录
            if batch_idx % self.log_interval == 0:
                self.writer.add_scalar(
                    "train/batch_loss", loss.item(),
                    self.current_epoch * len(self.train_loader) + batch_idx
                )
        
        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("train/epoch_loss", avg_loss, self.current_epoch)
        
        # 记录梯度范数
        if self.current_epoch % 5 == 0:
            self._log_gradient_norms(self.current_epoch)
            self._log_weight_histograms(self.current_epoch)
        
        # 记录当前学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)
        self.writer.add_scalar("train/learning_rate", current_lr, self.current_epoch)
        
        # 执行模型诊断（如果启用）
        if (self.enable_diagnostics and self.model_diagnostics and
            self.current_epoch % self.diagnostics_config.model_diagnosis_frequency == 0):
            self._perform_model_diagnosis(avg_loss)
        
        return avg_loss
    
    def _perform_model_diagnosis(self, train_loss):
        """执行模型诊断"""
        try:
            print(f"执行模型诊断 (epoch {self.current_epoch})...")
            
            # 创建诊断目录
            diag_dir = os.path.join(
                self.diagnostics_config.report_dir,
                f"epoch_{self.current_epoch:04d}"
            )
            os.makedirs(diag_dir, exist_ok=True)
            
            # 1. 分析梯度问题
            if self.diagnostics_config.check_gradients:
                # 需要重新计算损失以获取梯度
                self.model.train()
                sample_batch = next(iter(self.train_loader))
                low_dose = sample_batch[0].to(self.device)
                full_dose = sample_batch[1].to(self.device)
                
                self.optimizer.zero_grad()
                enhanced = self.model(low_dose)
                loss = self.criterion(enhanced, full_dose)
                
                gradient_report = self.model_diagnostics.analyze_gradients(
                    model=self.model,
                    loss=loss,
                    compute_norms=True,
                    detect_outliers=True
                )
                
                # 记录梯度问题
                if gradient_report.get('gradient_vanishing', False):
                    print(f"  警告: 检测到梯度消失 (总梯度范数: {gradient_report.get('total_l2_norm', 0):.6f})")
                if gradient_report.get('gradient_exploding', False):
                    print(f"  警告: 检测到梯度爆炸 (总梯度范数: {gradient_report.get('total_l2_norm', 0):.6f})")
                
                # 保存梯度报告
                gradient_report_path = os.path.join(diag_dir, "gradient_report.json")
                with open(gradient_report_path, 'w') as f:
                    import json
                    # 转换numpy类型为Python原生类型
                    def convert_to_serializable(obj):
                        if isinstance(obj, (np.integer, np.floating)):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, dict):
                            return {k: convert_to_serializable(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_to_serializable(item) for item in obj]
                        else:
                            return obj
                    
                    serializable_report = convert_to_serializable(gradient_report)
                    json.dump(serializable_report, f, indent=2)
                
                # 记录到TensorBoard
                if 'total_l2_norm' in gradient_report:
                    self.writer.add_scalar(
                        "diagnostics/gradient_total_norm",
                        gradient_report['total_l2_norm'],
                        self.current_epoch
                    )
                if 'zero_gradient_ratio' in gradient_report:
                    self.writer.add_scalar(
                        "diagnostics/gradient_zero_ratio",
                        gradient_report['zero_gradient_ratio'],
                        self.current_epoch
                    )
            
            # 2. 分析权重分布
            if self.diagnostics_config.check_weights:
                weight_report = self.model_diagnostics.analyze_weights(
                    model=self.model,
                    previous_weights=None,  # 可以扩展为跟踪权重变化
                    detect_anomalies=True
                )
                
                # 检查权重问题
                if 'weight_issues' in weight_report and weight_report['weight_issues']:
                    for issue in weight_report['weight_issues']:
                        print(f"  警告: {issue}")
                
                # 保存权重报告
                weight_report_path = os.path.join(diag_dir, "weight_report.json")
                with open(weight_report_path, 'w') as f:
                    import json
                    serializable_report = convert_to_serializable(weight_report)
                    json.dump(serializable_report, f, indent=2)
                
                # 记录到TensorBoard
                if 'global_stats' in weight_report:
                    stats = weight_report['global_stats']
                    if 'mean' in stats:
                        self.writer.add_scalar(
                            "diagnostics/weight_mean",
                            stats['mean'],
                            self.current_epoch
                        )
                    if 'std' in stats:
                        self.writer.add_scalar(
                            "diagnostics/weight_std",
                            stats['std'],
                            self.current_epoch
                        )
            
            # 3. 检测dead ReLU神经元
            if self.diagnostics_config.check_dead_relu:
                # 收集样本输入
                sample_inputs = []
                for i, (low_dose, _) in enumerate(self.train_loader):
                    if i >= 3:  # 使用3个批次
                        break
                    sample_inputs.append(low_dose.to(self.device))
                
                dead_relu_report = self.model_diagnostics.detect_dead_relu_neurons(
                    model=self.model,
                    sample_inputs=sample_inputs,
                    threshold=1e-6
                )
                
                # 检查dead ReLU问题
                if 'overall_dead_ratio' in dead_relu_report:
                    dead_ratio = dead_relu_report['overall_dead_ratio']
                    if dead_ratio > 0.3:
                        print(f"  警告: Dead ReLU比例较高 ({dead_ratio:.1%})")
                    else:
                        print(f"  Dead ReLU比例: {dead_ratio:.1%}")
                
                # 保存dead ReLU报告
                dead_relu_report_path = os.path.join(diag_dir, "dead_relu_report.json")
                with open(dead_relu_report_path, 'w') as f:
                    import json
                    serializable_report = convert_to_serializable(dead_relu_report)
                    json.dump(serializable_report, f, indent=2)
                
                # 记录到TensorBoard
                if 'overall_dead_ratio' in dead_relu_report:
                    self.writer.add_scalar(
                        "diagnostics/dead_relu_ratio",
                        dead_relu_report['overall_dead_ratio'],
                        self.current_epoch
                    )
            
            # 保存诊断历史
            self.model_diagnosis_history.append({
                'epoch': self.current_epoch,
                'timestamp': time.time(),
                'diagnosis_dir': diag_dir
            })
            
            print(f"模型诊断报告已保存到: {diag_dir}")
            
        except Exception as e:
            print(f"模型诊断失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _validate_batch_data(self, low_dose, full_dose, batch_idx):
        """
        验证批次数据的完整性
        
        参数:
            low_dose: 低剂量数据张量
            full_dose: 全剂量数据张量
            batch_idx: 批次索引
            
        返回:
            bool: 数据是否有效
            str: 错误信息（如果无效）
        """
        try:
            # 检查数据是否为None
            if low_dose is None or full_dose is None:
                return False, f"批次 {batch_idx}: 数据为None"
            
            # 检查数据类型
            if not isinstance(low_dose, torch.Tensor) or not isinstance(full_dose, torch.Tensor):
                return False, f"批次 {batch_idx}: 数据类型错误，期望torch.Tensor，实际类型: low_dose={type(low_dose)}, full_dose={type(full_dose)}"
            
            # 检查数据形状
            if low_dose.shape != full_dose.shape:
                return False, f"批次 {batch_idx}: 数据形状不匹配，low_dose={low_dose.shape}, full_dose={full_dose.shape}"
            
            # 检查数据维度
            if len(low_dose.shape) < 4:
                return False, f"批次 {batch_idx}: 数据维度不足，期望至少4维 (batch, channel, height, width)，实际形状: {low_dose.shape}"
            
            # 检查数据值范围
            if torch.all(low_dose == 0):
                return False, f"批次 {batch_idx}: 低剂量数据全为零"
            
            if torch.all(full_dose == 0):
                return False, f"批次 {batch_idx}: 全剂量数据全为零"
            
            # 检查NaN和Inf值
            if torch.any(torch.isnan(low_dose)):
                return False, f"批次 {batch_idx}: 低剂量数据包含NaN值"
            
            if torch.any(torch.isnan(full_dose)):
                return False, f"批次 {batch_idx}: 全剂量数据包含NaN值"
            
            if torch.any(torch.isinf(low_dose)):
                return False, f"批次 {batch_idx}: 低剂量数据包含Inf值"
            
            if torch.any(torch.isinf(full_dose)):
                return False, f"批次 {batch_idx}: 全剂量数据包含Inf值"
            
            # 检查数据范围是否合理
            low_min, low_max = low_dose.min().item(), low_dose.max().item()
            full_min, full_max = full_dose.min().item(), full_dose.max().item()
            
            if abs(low_max - low_min) < 1e-6:
                return False, f"批次 {batch_idx}: 低剂量数据范围过小 [{low_min:.6f}, {low_max:.6f}]"
            
            if abs(full_max - full_min) < 1e-6:
                return False, f"批次 {batch_idx}: 全剂量数据范围过小 [{full_min:.6f}, {full_max:.6f}]"
            
            return True, "数据验证通过"
            
        except Exception as e:
            return False, f"批次 {batch_idx}: 数据验证过程中发生异常: {type(e).__name__}: {str(e)}"
    
    def validate(self):
        """验证"""
        print(f"  开始验证阶段 (epoch {self.current_epoch})")
        
        # 检查验证数据加载器
        if self.val_loader is None:
            print("[ERROR] 验证数据加载器为None")
            return 0.0, 0.0, 0.0
        
        # 初始化变量，提供默认值
        val_dataset_size = 0
        val_batches = 0
        
        try:
            val_dataset_size = len(self.val_loader.dataset)
            val_batches = len(self.val_loader)
            print(f"  验证数据集大小: {val_dataset_size}")
            print(f"  验证批次数量: {val_batches}")
            print(f"  批次大小: {self.val_loader.batch_size}")
        except Exception as e:
            print(f"[ERROR] 无法获取验证数据加载器信息: {e}")
            # val_dataset_size和val_batches已经有默认值0，不会出现NameError
        
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        
        # 初始化统计计数器
        processed_batches = 0
        skipped_batches = 0
        
        # 存储批量数据用于可视化
        visualization_batch = None
        visualization_enhanced = None
        visualization_target = None
        
        try:
            with torch.no_grad():
                # 移除外层try-except，改为批次级别异常处理
                for batch_idx, (low_dose, full_dose) in enumerate(tqdm(self.val_loader, desc="验证")):
                    try:
                        # 使用数据完整性验证函数检查批次数据
                        is_valid, error_msg = self._validate_batch_data(low_dose, full_dose, batch_idx)
                        
                        if not is_valid:
                            print(f"[VALIDATION ERROR] 批次 {batch_idx} 数据验证失败: {error_msg}")
                            skipped_batches += 1
                            continue
                        
                        # 数据转移到设备
                        low_dose = low_dose.to(self.device)
                        full_dose = full_dose.to(self.device)
                        
                        # 模型前向传播
                        enhanced = self.model(low_dose)
                        
                        # 计算损失
                        loss = self.criterion(enhanced, full_dose)
                        
                        # 检查损失是否有效
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"[ERROR] 批次 {batch_idx}: 损失值为 {loss.item()}，使用默认值0.0")
                            loss_value = 0.0
                        else:
                            loss_value = loss.item()
                        
                        total_loss += loss_value
                        
                        # 计算指标
                        try:
                            psnr, ssim = calculate_metrics(enhanced, full_dose)
                            
                            # 检查指标是否有效
                            if psnr == 0 or ssim == 0:
                                print(f"[WARNING] 批次 {batch_idx}: PSNR={psnr:.4f}, SSIM={ssim:.4f}")
                            
                            total_psnr += psnr
                            total_ssim += ssim
                            
                        except Exception as metric_error:
                            print(f"[ERROR] 批次 {batch_idx}: 指标计算失败: {metric_error}")
                            # 使用默认指标值
                            psnr, ssim = 0.0, 0.0
                            total_psnr += psnr
                            total_ssim += ssim
                        
                        # 成功处理批次，增加计数器
                        processed_batches += 1
                        
                        # 收集第一个成功处理的批次用于可视化
                        if processed_batches == 1 and self.enable_diagnostics and self.visualizer:
                            visualization_batch = low_dose
                            visualization_enhanced = enhanced
                            visualization_target = full_dose
                            
                            # 打印第一个批次的详细信息
                            print(f"  第一个批次详细信息:")
                            print(f"  低剂量形状: {low_dose.shape}")
                            print(f"  全剂量形状: {full_dose.shape}")
                            print(f"  增强形状: {enhanced.shape}")
                            print(f"  低剂量范围: [{low_dose.min():.4f}, {low_dose.max():.4f}]")
                            print(f"  全剂量范围: [{full_dose.min():.4f}, {full_dose.max():.4f}]")
                            print(f"  增强范围: [{enhanced.min():.4f}, {enhanced.max():.4f}]")
                            print(f"  损失值: {loss_value:.6f}")
                            print(f"  PSNR: {psnr:.4f}")
                            print(f"  SSIM: {ssim:.4f}")
                            
                    except Exception as e:
                        print(f"[CRITICAL ERROR] 批次 {batch_idx} 处理发生致命异常: {type(e).__name__}: {str(e)}")
                        print(f"错误发生位置: 批次 {batch_idx}/{val_batches}")
                        print(f"已处理批次: {processed_batches}, 跳过批次: {skipped_batches}")
                        import traceback
                        traceback.print_exc()
                        
                        # 记录详细错误信息到TensorBoard
                        try:
                            self.writer.add_text("validation/batch_error",
                                               f"Epoch {self.current_epoch}, Batch {batch_idx}: {type(e).__name__}: {str(e)}",
                                               self.current_epoch)
                        except:
                            pass
                        
                        skipped_batches += 1
                        continue
        except Exception as e:
            # 打印详细的错误信息
            print(f"[CRITICAL] 验证循环发生致命错误: {type(e).__name__}: {str(e)}")
            print(f"错误发生位置: 验证循环外层 (数据加载器迭代或模型前向传播)")
            print(f"已处理批次: {processed_batches}, 跳过批次: {skipped_batches}")
            import traceback
            traceback.print_exc()
            
            # 尝试恢复而不是直接返回
            if processed_batches > 0:
                # 如果已经处理了部分批次，返回已计算的平均值
                avg_loss = total_loss / processed_batches
                avg_psnr = total_psnr / processed_batches
                avg_ssim = total_ssim / processed_batches
                print(f"  异常发生前已成功处理 {processed_batches} 个批次")
                print(f"  返回已计算的平均值: 损失={avg_loss:.6f}, PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}")
                
                # 记录错误信息到TensorBoard
                try:
                    self.writer.add_text("validation/error",
                                       f"Epoch {self.current_epoch}: {type(e).__name__}: {str(e)} - 已恢复处理",
                                       self.current_epoch)
                except:
                    pass
                    
                return avg_loss, avg_psnr, avg_ssim
            else:
                # 如果没有处理任何批次，返回合理的默认值
                print(f"  未成功处理任何批次，返回合理的默认值 (1.0, 0.0, 0.0)")
                
                # 记录详细错误信息到TensorBoard
                try:
                    self.writer.add_text("validation/error",
                                       f"Epoch {self.current_epoch}: {type(e).__name__}: {str(e)} - 无有效批次",
                                       self.current_epoch)
                except:
                    pass
                    
                return 1.0, 0.0, 0.0  # 返回合理的默认值
        
        # 检查是否有成功处理的批次
        if processed_batches == 0:
            print(f"[ERROR] 没有成功处理的批次 (总批次: {val_batches}, 跳过批次: {skipped_batches})")
            return 0.0, 0.0, 0.0
            
        # 使用实际处理批次数计算平均值
        avg_loss = total_loss / processed_batches
        avg_psnr = total_psnr / processed_batches
        avg_ssim = total_ssim / processed_batches
        
        print(f"  验证结果统计:")
        print(f"  总批次数量: {val_batches}")
        print(f"  实际处理批次: {processed_batches}")
        print(f"  跳过批次: {skipped_batches}")
        print(f"  总损失: {total_loss:.6f}")
        print(f"  总PSNR: {total_psnr:.4f}")
        print(f"  总SSIM: {total_ssim:.4f}")
        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  平均PSNR: {avg_psnr:.4f}")
        print(f"  平均SSIM: {avg_ssim:.4f}")
        
        # 检查平均值是否异常
        if avg_loss == 0:
            print("[WARNING] 平均损失为0")
        if avg_psnr == 0:
            print("[WARNING] 平均PSNR为0")
        if avg_ssim == 0:
            print("[WARNING] 平均SSIM为0")
        
        # TensorBoard记录
        try:
            self.writer.add_scalar("val/loss", avg_loss, self.current_epoch)
            self.writer.add_scalar("val/psnr", avg_psnr, self.current_epoch)
            self.writer.add_scalar("val/ssim", avg_ssim, self.current_epoch)
            print(f"  TensorBoard记录成功")
        except Exception as e:
            print(f"[ERROR] TensorBoard记录失败: {e}")
        
        # 执行验证集可视化（如果启用）
        if (self.enable_diagnostics and self.visualizer and visualization_batch is not None and
            self.current_epoch % self.diagnostics_config.visualization_frequency == 0):
            try:
                self._perform_validation_visualization(
                    visualization_batch, visualization_enhanced, visualization_target
                )
            except Exception as e:
                print(f"[ERROR] 验证可视化失败: {e}")
        
        # 计算详细指标（如果启用）
        if self.enable_diagnostics and self.metrics_calculator:
            try:
                self._calculate_detailed_metrics(visualization_enhanced, visualization_target)
            except Exception as e:
                print(f"[ERROR] 详细指标计算失败: {e}")
        
        print(f"  验证阶段完成 (处理批次: {processed_batches}/{val_batches})")
        return avg_loss, avg_psnr, avg_ssim
    
    def _perform_validation_visualization(self, low_dose, enhanced, full_dose):
        """执行验证集可视化"""
        try:
            print(f"执行验证集可视化 (epoch {self.current_epoch})...")
            
            # 创建可视化目录
            vis_dir = os.path.join(
                self.diagnostics_config.visualization_dir,
                f"epoch_{self.current_epoch:04d}"
            )
            os.makedirs(vis_dir, exist_ok=True)
            
            # 生成可视化
            figures = self.visualizer.visualize_batch(
                low_dose_batch=low_dose,
                enhanced_batch=enhanced,
                full_dose_batch=full_dose,
                max_samples=self.diagnostics_config.visualize_samples,
                save_dir=vis_dir,
                prefix=f"epoch_{self.current_epoch:04d}"
            )
            
            # 记录到TensorBoard
            if figures and len(figures) > 0:
                try:
                    import matplotlib.pyplot as plt
                    
                    fig = figures[0]
                    # 使用add_figure方法直接记录matplotlib图形
                    self.writer.add_figure(
                        "validation/visualization",
                        fig,
                        self.current_epoch,
                        close=True  # 自动关闭图形以释放内存
                    )
                except (ImportError, AttributeError) as e:
                    print(f"警告: 无法将可视化图像记录到TensorBoard: {e}")
                    # 如果add_figure不可用，尝试回退到手动关闭图形
                    try:
                        import matplotlib.pyplot as plt
                        plt.close(fig)
                    except:
                        pass
            
            print(f"验证集可视化已保存到: {vis_dir}")
            
        except Exception as e:
            print(f"验证集可视化失败: {e}")
    
    def _calculate_detailed_metrics(self, enhanced, full_dose):
        """计算详细指标"""
        try:
            if enhanced is None or full_dose is None:
                return
            
            print(f"计算详细指标 (epoch {self.current_epoch})...")
            
            # 计算详细指标
            detailed_metrics = self.metrics_calculator.calculate_all_metrics_batch(
                pred=enhanced,
                target=full_dose,
                data_range=1.0,
                use_gpu=(self.device.type == 'cuda')
            )
            
            # 记录到TensorBoard
            for metric_name, metric_value in detailed_metrics.items():
                if isinstance(metric_value, (int, float)):
                    self.writer.add_scalar(
                        f"val/detailed/{metric_name}",
                        metric_value,
                        self.current_epoch
                    )
            
            # 保存到历史记录
            self.detailed_metrics_history.append({
                'epoch': self.current_epoch,
                'metrics': detailed_metrics
            })
            
            # 打印关键指标
            if 'rmse' in detailed_metrics:
                print(f"  RMSE: {detailed_metrics['rmse']:.4f} ± {detailed_metrics.get('rmse_std', 0):.4f}")
            if 'mae' in detailed_metrics:
                print(f"  MAE: {detailed_metrics['mae']:.4f} ± {detailed_metrics.get('mae_std', 0):.4f}")
            if 'psnr' in detailed_metrics:
                print(f"  PSNR: {detailed_metrics['psnr']:.2f} dB ± {detailed_metrics.get('psnr_std', 0):.2f}")
            if 'ssim' in detailed_metrics:
                print(f"  SSIM: {detailed_metrics['ssim']:.4f} ± {detailed_metrics.get('ssim_std', 0):.4f}")
            
        except Exception as e:
            print(f"详细指标计算失败: {e}")
    
    def train(self) -> None:
        """主训练循环"""
        print("开始训练...")
        start_time = time.time()
        
        # 在训练开始时添加模型输出范围检查
        self._check_model_output_range()
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # 添加梯度范数监控
            if epoch % 10 == 0:  # 每10个epoch监控一次
                self._monitor_gradient_norms()
            
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, val_psnr, val_ssim = self.validate()
            self.val_losses.append(val_loss)
            self.val_psnrs.append(val_psnr)
            self.val_ssims.append(val_ssim)
            
            # 学习率调度
            if self.warmup_scheduler and epoch < self.warmup_epochs:
                # 热身调度器：在optimizer.step()之后调用
                self.warmup_scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"热身阶段 Epoch {epoch+1}/{self.warmup_epochs}, 学习率: {current_lr:.6f}")
            elif self.scheduler:
                # 主调度器：热身结束后使用
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 打印进度
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{self.config.training.num_epochs}: "
                  f"训练损失={train_loss:.4f}, "
                  f"验证损失={val_loss:.4f}, "
                  f"PSNR={val_psnr:.2f} dB, "
                  f"SSIM={val_ssim:.4f}, "
                  f"学习率={current_lr:.6f}")
            
            # 执行训练曲线分析（如果启用）
            if (self.enable_diagnostics and self.training_analyzer and
                epoch % self.diagnostics_config.training_analysis_frequency == 0 and
                len(self.train_losses) >= 5):  # 至少有5个epoch的数据
                self._perform_training_curve_analysis()
            
            # 检查早停
            if self._check_early_stopping(val_loss):
                print(f"早停在 epoch {epoch+1} 触发")
                break
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epoch, val_loss,
                    os.path.join(self.config.training.checkpoint_dir, "best_model.pth"),
                    additional_info={
                        'train_loss': train_loss,
                        'val_psnr': val_psnr,
                        'val_ssim': val_ssim
                    }
                )
                print(f"保存最佳模型，验证损失: {val_loss:.4f}")
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epoch, val_loss,
                    os.path.join(self.config.training.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                )
        
        # 训练完成
        training_time = time.time() - start_time
        print(f"训练完成！总时间: {training_time:.2f}秒")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"最佳PSNR: {max(self.val_psnrs) if self.val_psnrs else 0:.2f} dB")
        print(f"最佳SSIM: {max(self.val_ssims) if self.val_ssims else 0:.4f}")
        
        # 保存最终模型
        save_checkpoint(
            self.model, self.optimizer, self.scheduler, epoch, val_loss,
            os.path.join(self.config.training.checkpoint_dir, "final_model.pth")
        )
        
        # 保存训练历史
        self._save_training_history()
        
        self.writer.close()
        
        return self.train_losses, self.val_losses, self.val_psnrs, self.val_ssims
    
    def _perform_training_curve_analysis(self):
        """执行训练曲线分析"""
        try:
            if len(self.train_losses) < 5 or len(self.val_losses) < 5:
                return
            
            print(f"执行训练曲线分析 (epoch {self.current_epoch})...")
            
            # 分析过拟合/欠拟合
            overfitting_report = self.training_analyzer.analyze_overfitting(
                train_losses=self.train_losses,
                val_losses=self.val_losses,
                epochs=list(range(1, len(self.train_losses) + 1))
            )
            
            # 打印分析结果
            if overfitting_report.get('is_overfitting', False):
                print(f"  警告: 检测到过拟合 (损失比率: {overfitting_report.get('loss_ratio', 0):.2f})")
                print(f"  建议: 增加正则化、早停、数据增强或减少模型复杂度")
            elif overfitting_report.get('is_underfitting', False):
                print(f"  警告: 检测到欠拟合 (训练损失: {overfitting_report.get('train_loss_final', 0):.4f})")
                print(f"  建议: 增加模型容量、训练更长时间、减少正则化")
            else:
                print(f"  训练状态: 正常 (损失比率: {overfitting_report.get('loss_ratio', 0):.2f})")
            
            # 记录到TensorBoard
            if 'loss_ratio' in overfitting_report:
                self.writer.add_scalar(
                    "diagnostics/loss_ratio",
                    overfitting_report['loss_ratio'],
                    self.current_epoch
                )
            
            if 'overfitting_gap' in overfitting_report:
                self.writer.add_scalar(
                    "diagnostics/overfitting_gap",
                    overfitting_report['overfitting_gap'],
                    self.current_epoch
                )
            
            # 分析学习率
            if self.diagnostics_config.check_learning_rate and len(self.learning_rates) >= 5:
                lr_analysis = self._analyze_learning_rate()
                if lr_analysis:
                    print(f"  学习率分析: {lr_analysis}")
            
            # 保存分析报告
            analysis_dir = os.path.join(
                self.diagnostics_config.report_dir,
                f"epoch_{self.current_epoch:04d}"
            )
            os.makedirs(analysis_dir, exist_ok=True)
            
            analysis_report = {
                'epoch': self.current_epoch,
                'timestamp': time.time(),
                'num_epochs_analyzed': len(self.train_losses),
                'overfitting_analysis': overfitting_report,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_psnrs': self.val_psnrs,
                'val_ssims': self.val_ssims,
                'learning_rates': self.learning_rates
            }
            
            analysis_report_path = os.path.join(analysis_dir, "training_analysis.json")
            with open(analysis_report_path, 'w') as f:
                import json
                json.dump(analysis_report, f, indent=2)
            
            # 保存到历史记录
            self.training_analysis_history.append({
                'epoch': self.current_epoch,
                'analysis': overfitting_report,
                'report_path': analysis_report_path
            })
            
            print(f"训练曲线分析报告已保存到: {analysis_report_path}")
            
        except Exception as e:
            print(f"训练曲线分析失败: {e}")
    
    def _analyze_learning_rate(self):
        """分析学习率是否合适"""
        if len(self.learning_rates) < 5:
            return None
        
        current_lr = self.learning_rates[-1]
        avg_lr = sum(self.learning_rates) / len(self.learning_rates)
        
        # 检查学习率是否过小
        if current_lr < 1e-6:
            return "学习率过小，可能收敛缓慢"
        
        # 检查学习率是否过大
        if current_lr > 1e-2:
            return "学习率过大，可能导致训练不稳定"
        
        # 检查学习率下降是否过快
        if len(self.learning_rates) >= 10:
            recent_lrs = self.learning_rates[-10:]
            lr_decline = (recent_lrs[0] - recent_lrs[-1]) / recent_lrs[0]
            if lr_decline > 0.9:
                return "学习率下降过快，可能过早停止学习"
        
        return f"学习率合适 ({current_lr:.6f})"
    
    def _save_training_history(self):
        """保存训练历史到文件"""
        import pandas as pd
        
        history = {
            'epoch': list(range(1, len(self.train_losses) + 1)),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'val_psnr': self.val_psnrs,
            'val_ssim': self.val_ssims,
            'learning_rate': self.learning_rates[:len(self.train_losses)]
        }
        
        df = pd.DataFrame(history)
        history_path = os.path.join(self.config.training.checkpoint_dir, "training_history.csv")
        df.to_csv(history_path, index=False)
        print(f"训练历史已保存到: {history_path}")
        
        # 绘制训练曲线
        self._plot_training_curves(df)
    
    def _plot_training_curves(self, df):
        """绘制训练曲线"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # 损失曲线
            axes[0, 0].plot(df['epoch'], df['train_loss'], label='训练损失')
            axes[0, 0].plot(df['epoch'], df['val_loss'], label='验证损失')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('损失')
            axes[0, 0].set_title('训练和验证损失')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # PSNR曲线
            axes[0, 1].plot(df['epoch'], df['val_psnr'], label='验证PSNR', color='green')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('PSNR (dB)')
            axes[0, 1].set_title('验证PSNR')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # SSIM曲线
            axes[1, 0].plot(df['epoch'], df['val_ssim'], label='验证SSIM', color='orange')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('SSIM')
            axes[1, 0].set_title('验证SSIM')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 学习率曲线
            axes[1, 1].plot(df['epoch'], df['learning_rate'], label='学习率', color='red')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('学习率')
            axes[1, 1].set_title('学习率变化')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(self.config.training.checkpoint_dir, "training_curves.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            
            print(f"训练曲线图已保存到: {plot_path}")
            
        except ImportError:
            print("警告: 未安装matplotlib，无法绘制训练曲线")
    
    def _check_model_output_range(self):
        """
        检查模型输出范围
        
        在训练开始时验证模型输出是否在合理范围内
        """
        print("检查模型输出范围...")
        self.model.eval()
        
        try:
            # 获取一个批次的数据
            low_dose, full_dose = next(iter(self.train_loader))
            low_dose = low_dose.to(self.device)
            
            with torch.no_grad():
                # 前向传播
                output = self.model(low_dose)
                
                # 检查输出范围
                min_val = output.min().item()
                max_val = output.max().item()
                mean_val = output.mean().item()
                std_val = output.std().item()
                
                print(f"  模型输出统计:")
                print(f"    最小值: {min_val:.6f}")
                print(f"    最大值: {max_val:.6f}")
                print(f"    平均值: {mean_val:.6f}")
                print(f"    标准差: {std_val:.6f}")
                
                # 检查是否有异常值
                if torch.isnan(output).any():
                    print("  ⚠️ 警告: 模型输出包含NaN值!")
                if torch.isinf(output).any():
                    print("  ⚠️ 警告: 模型输出包含无穷大值!")
                
                # 检查输出是否在合理范围内（对于CT图像，通常在[-1000, 1000] HU范围内）
                if abs(min_val) > 2000 or abs(max_val) > 2000:
                    print("  ⚠️ 警告: 模型输出范围异常，可能超出CT图像典型范围")
                else:
                    print("  ✓ 模型输出范围正常")
                    
        except Exception as e:
            print(f"  模型输出范围检查失败: {e}")
        
        self.model.train()
    
    def _monitor_gradient_norms(self):
        """
        监控梯度范数
        
        计算并记录模型参数的梯度范数
        """
        print(f"监控梯度范数 (Epoch {self.current_epoch})...")
        
        total_norm = 0.0
        param_count = 0
        gradient_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # 计算该参数的梯度范数
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
                
                # 记录各层梯度统计
                layer_name = name.split('.')[0] if '.' in name else name
                if layer_name not in gradient_stats:
                    gradient_stats[layer_name] = {
                        'count': 0,
                        'sum_norm': 0.0,
                        'max_norm': 0.0
                    }
                
                gradient_stats[layer_name]['count'] += 1
                gradient_stats[layer_name]['sum_norm'] += param_norm
                gradient_stats[layer_name]['max_norm'] = max(
                    gradient_stats[layer_name]['max_norm'], param_norm
                )
        
        if param_count > 0:
            # 计算总梯度范数
            total_norm = total_norm ** 0.5
            avg_norm = total_norm / param_count
            
            print(f"  总梯度范数: {total_norm:.6f}")
            print(f"  平均梯度范数: {avg_norm:.6f}")
            print(f"  有梯度的参数数量: {param_count}")
            
            # 记录到TensorBoard
            if hasattr(self, 'writer'):
                self.writer.add_scalar('gradients/total_norm', total_norm, self.current_epoch)
                self.writer.add_scalar('gradients/avg_norm', avg_norm, self.current_epoch)
            
            # 检查梯度消失/爆炸
            if total_norm < 1e-7:
                print("  ⚠️ 警告: 梯度可能消失 (总范数 < 1e-7)")
            elif total_norm > 1000:
                print("  ⚠️ 警告: 梯度可能爆炸 (总范数 > 1000)")
            else:
                print("  ✓ 梯度范数正常")
            
            # 打印各层梯度统计
            if len(gradient_stats) <= 10:  # 避免输出过多
                print("  各层梯度统计:")
                for layer_name, stats in gradient_stats.items():
                    avg_layer_norm = stats['sum_norm'] / stats['count'] if stats['count'] > 0 else 0
                    print(f"    {layer_name}: 平均={avg_layer_norm:.6f}, 最大={stats['max_norm']:.6f}")
        else:
            print("  没有找到梯度信息")
    
    def test(self):
        """在测试集上评估"""
        print("测试...")
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        
        with torch.no_grad():
            for low_dose, full_dose in tqdm(self.test_loader, desc="测试"):
                low_dose = low_dose.to(self.device)
                full_dose = full_dose.to(self.device)
                
                enhanced = self.model(low_dose)
                loss = self.criterion(enhanced, full_dose)
                total_loss += loss.item()
                
                psnr, ssim = calculate_metrics(enhanced, full_dose)
                total_psnr += psnr
                total_ssim += ssim
        
        avg_loss = total_loss / len(self.test_loader)
        avg_psnr = total_psnr / len(self.test_loader)
        avg_ssim = total_ssim / len(self.test_loader)
        
        print(f"测试结果: 损失={avg_loss:.4f}, PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}")
        
        return avg_loss, avg_psnr, avg_ssim
    
    # ===== 诊断配置控制方法 =====
    
    def set_diagnostics_enabled(self, enable: bool = True):
        """启用或禁用诊断功能"""
        self.enable_diagnostics = enable
        if enable and not hasattr(self, 'metrics_calculator'):
            # 重新初始化诊断工具
            self._initialize_diagnostics()
        print(f"诊断功能已{'启用' if enable else '禁用'}")
    
    def _initialize_diagnostics(self):
        """初始化诊断工具（内部方法）"""
        if not hasattr(self.config, 'diagnostics'):
            print("警告: 配置中没有诊断配置，使用默认配置")
            from ..Tools.diagnostics import DiagnosticsConfig
            self.diagnostics_config = DiagnosticsConfig()
        else:
            self.diagnostics_config = DiagnosticsConfig(
                enable_diagnostics=self.config.diagnostics.enable_diagnostics,
                compute_rmse=getattr(self.config.diagnostics, 'compute_rmse', True),
                compute_mae=getattr(self.config.diagnostics, 'compute_mae', True),
                compute_psnr=getattr(self.config.diagnostics, 'compute_psnr', True),
                compute_ssim=getattr(self.config.diagnostics, 'compute_ssim', True),
                compute_ms_ssim=getattr(self.config.diagnostics, 'compute_ms_ssim', False),
                compute_lpips=getattr(self.config.diagnostics, 'compute_lpips', False),
                visualize_samples=getattr(self.config.diagnostics, 'visualize_samples', 5),
                save_visualizations=getattr(self.config.diagnostics, 'save_visualizations', True),
                visualization_dir=getattr(self.config.diagnostics, 'visualization_dir', "./diagnostics/visualizations"),
                dpi=getattr(self.config.diagnostics, 'dpi', 150),
                visualization_frequency=getattr(self.config.diagnostics, 'visualization_frequency', 5),
                check_gradients=getattr(self.config.diagnostics, 'check_gradients', True),
                check_weights=getattr(self.config.diagnostics, 'check_weights', True),
                check_activations=getattr(self.config.diagnostics, 'check_activations', False),
                check_dead_relu=getattr(self.config.diagnostics, 'check_dead_relu', True),
                model_diagnosis_frequency=getattr(self.config.diagnostics, 'model_diagnosis_frequency', 10),
                analyze_overfitting=getattr(self.config.diagnostics, 'analyze_overfitting', True),
                compute_loss_ratio=getattr(self.config.diagnostics, 'compute_loss_ratio', True),
                check_learning_rate=getattr(self.config.diagnostics, 'check_learning_rate', True),
                training_analysis_frequency=getattr(self.config.diagnostics, 'training_analysis_frequency', 5),
                generate_html_report=getattr(self.config.diagnostics, 'generate_html_report', True),
                generate_pdf_report=getattr(self.config.diagnostics, 'generate_pdf_report', False),
                report_dir=getattr(self.config.diagnostics, 'report_dir', "./diagnostics/reports")
            )
        
        # 初始化诊断工具
        self.metrics_calculator = ImageMetricsCalculator(self.diagnostics_config)
        self.visualizer = ValidationVisualizer(self.diagnostics_config)
        self.model_diagnostics = ModelDiagnostics(self.diagnostics_config)
        self.training_analyzer = TrainingCurveAnalyzer(self.diagnostics_config)
        
        # 诊断状态
        self.detailed_metrics_history = []
        self.model_diagnosis_history = []
        self.training_analysis_history = []
    
    def update_diagnostics_config(self, **kwargs):
        """更新诊断配置"""
        if not hasattr(self, 'diagnostics_config') or self.diagnostics_config is None:
            print("错误: 诊断配置未初始化")
            return
        
        for key, value in kwargs.items():
            if hasattr(self.diagnostics_config, key):
                setattr(self.diagnostics_config, key, value)
                print(f"更新诊断配置: {key} = {value}")
            else:
                print(f"警告: 诊断配置中没有属性 '{key}'")
    
    def get_diagnostics_summary(self):
        """获取诊断功能摘要"""
        if not self.enable_diagnostics:
            return {"diagnostics_enabled": False}
        
        summary = {
            "diagnostics_enabled": True,
            "metrics_calculator": self.metrics_calculator is not None,
            "visualizer": self.visualizer is not None,
            "model_diagnostics": self.model_diagnostics is not None,
            "training_analyzer": self.training_analyzer is not None,
            "config": {
                "visualization_frequency": self.diagnostics_config.visualization_frequency,
                "model_diagnosis_frequency": self.diagnostics_config.model_diagnosis_frequency,
                "training_analysis_frequency": self.diagnostics_config.training_analysis_frequency,
                "compute_rmse": self.diagnostics_config.compute_rmse,
                "compute_mae": self.diagnostics_config.compute_mae,
                "compute_psnr": self.diagnostics_config.compute_psnr,
                "compute_ssim": self.diagnostics_config.compute_ssim,
                "check_gradients": self.diagnostics_config.check_gradients,
                "check_weights": self.diagnostics_config.check_weights,
                "check_dead_relu": self.diagnostics_config.check_dead_relu,
                "analyze_overfitting": self.diagnostics_config.analyze_overfitting
            },
            "history": {
                "detailed_metrics": len(self.detailed_metrics_history),
                "model_diagnosis": len(self.model_diagnosis_history),
                "training_analysis": len(self.training_analysis_history)
            }
        }
        return summary
    
    def generate_diagnostics_report(self, output_dir: Optional[str] = None):
        """生成诊断报告"""
        if not self.enable_diagnostics:
            print("错误: 诊断功能未启用")
            return
        
        try:
            from ..Tools.diagnostics import DiagnosticsCLI
            import json
            
            if output_dir is None:
                output_dir = self.diagnostics_config.report_dir
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 收集诊断数据
            diagnostic_data = {
                "training_summary": {
                    "total_epochs": len(self.train_losses),
                    "best_val_loss": self.best_val_loss,
                    "best_val_psnr": max(self.val_psnrs) if self.val_psnrs else 0,
                    "best_val_ssim": max(self.val_ssims) if self.val_ssims else 0
                },
                "detailed_metrics": self.detailed_metrics_history,
                "model_diagnosis": self.model_diagnosis_history,
                "training_analysis": self.training_analysis_history,
                "config": self.get_diagnostics_summary()["config"]
            }
            
            # 保存为JSON
            report_path = os.path.join(output_dir, "diagnostics_report.json")
            with open(report_path, 'w') as f:
                json.dump(diagnostic_data, f, indent=2, default=str)
            
            print(f"诊断报告已保存到: {report_path}")
            
            # 如果启用HTML报告，生成HTML
            if self.diagnostics_config.generate_html_report:
                try:
                    cli = DiagnosticsCLI()
                    html_report = cli.generate_html_report(diagnostic_data)
                    html_path = os.path.join(output_dir, "diagnostics_report.html")
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(html_report)
                    print(f"HTML诊断报告已保存到: {html_path}")
                except Exception as e:
                    print(f"生成HTML报告失败: {e}")
            
            return report_path
            
        except Exception as e:
            print(f"生成诊断报告失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== 性能优化方法 =====
    
    def optimize_diagnostics_performance(self):
        """优化诊断性能"""
        if not self.enable_diagnostics:
            return
        
        print("优化诊断性能...")
        
        # 1. 调整诊断频率以避免影响训练速度
        if len(self.train_losses) > 50:  # 训练后期减少诊断频率
            if self.diagnostics_config.visualization_frequency < 10:
                self.diagnostics_config.visualization_frequency = 10
                print(f"  调整可视化频率为每 {self.diagnostics_config.visualization_frequency} 个epoch")
            
            if self.diagnostics_config.model_diagnosis_frequency < 20:
                self.diagnostics_config.model_diagnosis_frequency = 20
                print(f"  调整模型诊断频率为每 {self.diagnostics_config.model_diagnosis_frequency} 个epoch")
        
        # 2. 禁用计算成本高的指标（如果训练速度过慢）
        if hasattr(self, 'train_losses') and len(self.train_losses) > 0:
            avg_epoch_time = self._estimate_epoch_time()
            if avg_epoch_time > 300:  # 如果每个epoch超过5分钟
                if self.diagnostics_config.compute_ms_ssim:
                    self.diagnostics_config.compute_ms_ssim = False
                    print("  禁用MS-SSIM计算（计算成本高）")
                
                if self.diagnostics_config.compute_lpips:
                    self.diagnostics_config.compute_lpips = False
                    print("  禁用LPIPS计算（计算成本高）")
                
                if self.diagnostics_config.check_activations:
                    self.diagnostics_config.check_activations = False
                    print("  禁用激活值检查（影响性能）")
        
        # 3. 减少可视化样本数量
        if self.diagnostics_config.visualize_samples > 3:
            self.diagnostics_config.visualize_samples = 3
            print(f"  减少可视化样本数量为 {self.diagnostics_config.visualize_samples}")
    
    def _estimate_epoch_time(self):
        """估计每个epoch的平均训练时间"""
        if not hasattr(self, '_epoch_times') or len(self._epoch_times) < 3:
            return 0
        
        # 计算最近几个epoch的平均时间
        recent_times = self._epoch_times[-5:] if len(self._epoch_times) >= 5 else self._epoch_times
        return sum(recent_times) / len(recent_times)
    
    def enable_performance_mode(self, enable: bool = True):
        """启用性能模式（减少诊断开销）"""
        if not self.enable_diagnostics:
            return
        
        if enable:
            print("启用诊断性能模式...")
            # 保存原始配置
            if not hasattr(self, '_original_diagnostics_config'):
                self._original_diagnostics_config = {
                    'visualization_frequency': self.diagnostics_config.visualization_frequency,
                    'model_diagnosis_frequency': self.diagnostics_config.model_diagnosis_frequency,
                    'training_analysis_frequency': self.diagnostics_config.training_analysis_frequency,
                    'compute_ms_ssim': self.diagnostics_config.compute_ms_ssim,
                    'compute_lpips': self.diagnostics_config.compute_lpips,
                    'check_activations': self.diagnostics_config.check_activations,
                    'visualize_samples': self.diagnostics_config.visualize_samples
                }
            
            # 应用性能优化配置
            self.diagnostics_config.visualization_frequency = max(10, self.diagnostics_config.visualization_frequency)
            self.diagnostics_config.model_diagnosis_frequency = max(20, self.diagnostics_config.model_diagnosis_frequency)
            self.diagnostics_config.training_analysis_frequency = max(10, self.diagnostics_config.training_analysis_frequency)
            self.diagnostics_config.compute_ms_ssim = False
            self.diagnostics_config.compute_lpips = False
            self.diagnostics_config.check_activations = False
            self.diagnostics_config.visualize_samples = min(3, self.diagnostics_config.visualize_samples)
            
            print("  性能模式已启用：减少诊断频率，禁用高成本计算")
        else:
            # 恢复原始配置
            if hasattr(self, '_original_diagnostics_config'):
                print("恢复原始诊断配置...")
                for key, value in self._original_diagnostics_config.items():
                    setattr(self.diagnostics_config, key, value)
                delattr(self, '_original_diagnostics_config')
                print("  原始配置已恢复")
    
    def measure_diagnostics_overhead(self):
        """测量诊断功能的时间开销"""
        if not self.enable_diagnostics:
            return {"diagnostics_enabled": False}
        
        import time
        overhead_report = {
            "diagnostics_enabled": True,
            "metrics_calculation_time": 0,
            "visualization_time": 0,
            "model_diagnosis_time": 0,
            "training_analysis_time": 0,
            "total_overhead_per_epoch": 0
        }
        
        # 这里可以添加实际测量代码
        # 在实际实现中，可以记录每个诊断功能的时间开销
        
        return overhead_report
    
    def print_performance_tips(self):
        """打印性能优化建议"""
        print("=" * 60)
        print("诊断性能优化建议:")
        print("=" * 60)
        print("1. 调整诊断频率:")
        print("   - 可视化频率: 每 {} 个epoch".format(self.diagnostics_config.visualization_frequency))
        print("   - 模型诊断频率: 每 {} 个epoch".format(self.diagnostics_config.model_diagnosis_frequency))
        print("   - 训练分析频率: 每 {} 个epoch".format(self.diagnostics_config.training_analysis_frequency))
        print()
        print("2. 禁用高成本计算:")
        print("   - MS-SSIM计算: {}".format("启用" if self.diagnostics_config.compute_ms_ssim else "禁用"))
        print("   - LPIPS计算: {}".format("启用" if self.diagnostics_config.compute_lpips else "禁用"))
        print("   - 激活值检查: {}".format("启用" if self.diagnostics_config.check_activations else "禁用"))
        print()
        print("3. 资源使用:")
        print("   - 可视化样本数量: {}".format(self.diagnostics_config.visualize_samples))
        print()
        print("4. 性能模式:")
        print("   - 使用 trainer.enable_performance_mode(True) 启用性能模式")
        print("   - 使用 trainer.optimize_diagnostics_performance() 自动优化")
        print("=" * 60)


def main():
    """主函数"""
    # 加载配置
    config = Config()
    
    # 如果没有数据，创建虚拟数据
    if not os.path.exists(config.data.data_dir) or \
       not os.listdir(os.path.join(config.data.data_dir, config.data.low_dose_dir)):
        print("未找到数据，创建虚拟数据...")
        create_dummy_data(config.data, num_samples=50)
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 训练
    trainer.train()
    
    # 测试
    trainer.test()


if __name__ == "__main__":
    main()
