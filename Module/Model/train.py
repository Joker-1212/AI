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
        
        print(f"设备: {self.device}")
        print(f"模型: {config.model.model_name}")
        print(f"参数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
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
        if self.config.training.scheduler.lower() == "reducelronplateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.training.patience // 2,
                min_lr=self.config.training.min_lr
            )
        elif self.config.training.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.min_lr
            )
        elif self.config.training.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.5
            )
        else:
            return None
    
    def _create_loss_function(self):
        """创建损失函数"""
        if self.config.training.loss_function.lower() == "l1loss":
            return nn.L1Loss()
        elif self.config.training.loss_function.lower() == "mseloss":
            return nn.MSELoss()
        elif self.config.training.loss_function.lower() == "smoothl1":
            return nn.SmoothL1Loss()
        elif self.config.training.loss_function.lower() == "ssimloss":
            # 自定义SSIM损失
            from monai.losses import SSIMLoss
            return SSIMLoss(spatial_dims=3)
        else:
            raise ValueError(f"未知损失函数: {self.config.training.loss_function}")
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"训练 Epoch {self.current_epoch}")
        
        for batch_idx, (low_dose, full_dose) in enumerate(progress_bar):
            low_dose = low_dose.to(self.device)
            full_dose = full_dose.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            enhanced = self.model(low_dose)
            # RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 1 but got size 2 for tensor number 1 in the list.
            loss = self.criterion(enhanced, full_dose)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            # TensorBoard记录
            if batch_idx % 10 == 0:
                self.writer.add_scalar(
                    "train/batch_loss", loss.item(),
                    self.current_epoch * len(self.train_loader) + batch_idx
                )
        
        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("train/epoch_loss", avg_loss, self.current_epoch)
        
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
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss, val_psnr, val_ssim = self.validate()
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 打印进度
            print(f"Epoch {epoch+1}/{self.config.training.num_epochs}: "
                  f"训练损失={train_loss:.4f}, "
                  f"验证损失={val_loss:.4f}, "
                  f"PSNR={val_psnr:.2f} dB, "
                  f"SSIM={val_ssim:.4f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epoch, val_loss,
                    os.path.join(self.config.training.checkpoint_dir, "best_model.pth")
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
        
        # 保存最终模型
        save_checkpoint(
            self.model, self.optimizer, self.scheduler, epoch, val_loss,
            os.path.join(self.config.training.checkpoint_dir, "final_model.pth")
        )
        
        self.writer.close()
    
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
