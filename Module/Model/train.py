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

from ..Config.config import Config
from ..Loader.data_loader import prepare_data_loaders, create_dummy_data
from .models import create_model
from ..Tools.utils import save_checkpoint, load_checkpoint, calculate_metrics


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
            self.scaler = torch.cuda.amp.GradScaler()
        
        # 日志和监控
        self.log_interval = getattr(config.training, 'log_interval', 10)
        self.save_interval = getattr(config.training, 'save_interval', 10)
        
        # 训练监控
        self.train_losses = []
        self.val_losses = []
        self.val_psnrs = []
        self.val_ssims = []
        self.learning_rates = []
        
        print(f"设备: {self.device}")
        print(f"模型: {config.model.model_name}")
        print(f"参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"优化器: {config.training.optimizer}")
        print(f"损失函数: {config.training.loss_function}")
        if self.config.training.use_multi_scale_loss:
            print(f"使用多尺度损失，权重: {self.config.training.multi_scale_weights}")
    
    def _create_optimizer(self):
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
    
    def _create_scheduler(self):
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
        
        return avg_loss
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        
        with torch.no_grad():
            for low_dose, full_dose in tqdm(self.val_loader, desc="验证"):
                low_dose = low_dose.to(self.device)
                full_dose = full_dose.to(self.device)
                
                enhanced = self.model(low_dose)
                loss = self.criterion(enhanced, full_dose)
                total_loss += loss.item()
                
                # 计算指标
                psnr, ssim = calculate_metrics(enhanced, full_dose)
                total_psnr += psnr
                total_ssim += ssim
        
        avg_loss = total_loss / len(self.val_loader)
        avg_psnr = total_psnr / len(self.val_loader)
        avg_ssim = total_ssim / len(self.val_loader)
        
        # TensorBoard记录
        self.writer.add_scalar("val/loss", avg_loss, self.current_epoch)
        self.writer.add_scalar("val/psnr", avg_psnr, self.current_epoch)
        self.writer.add_scalar("val/ssim", avg_ssim, self.current_epoch)
        
        return avg_loss, avg_psnr, avg_ssim
    
    def train(self):
        """主训练循环"""
        print("开始训练...")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # 学习率热身
            if self.warmup_scheduler and epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"热身阶段 Epoch {epoch+1}/{self.warmup_epochs}, 学习率: {current_lr:.6f}")
            
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, val_psnr, val_ssim = self.validate()
            self.val_losses.append(val_loss)
            self.val_psnrs.append(val_psnr)
            self.val_ssims.append(val_ssim)
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif epoch >= self.warmup_epochs:  # 热身结束后才使用主调度器
                    self.scheduler.step()
            
            # 打印进度
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{self.config.training.num_epochs}: "
                  f"训练损失={train_loss:.4f}, "
                  f"验证损失={val_loss:.4f}, "
                  f"PSNR={val_psnr:.2f} dB, "
                  f"SSIM={val_ssim:.4f}, "
                  f"学习率={current_lr:.6f}")
            
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
