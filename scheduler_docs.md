# PyTorch 学习率调度器详细说明

学习率调度器是深度学习训练中的关键组件，用于在训练过程中动态调整学习率，以改善模型收敛性和性能。PyTorch提供了多种学习率调度器，每种都有不同的调度策略和适用场景。

## 1. 基础调度器

### 1.1 StepLR（步长调度器）
**调度方式**：每隔固定步数（epoch）将学习率乘以一个衰减因子
**公式**：`lr = lr * gamma^(epoch // step_size)`
**主要参数**：
- `step_size`：学习率衰减的步长（epoch数）
- `gamma`：衰减因子（默认0.1）
**适用场景**：简单的周期性学习率衰减，训练过程稳定时使用
**配置示例**：
```yaml
scheduler: "StepLR"
scheduler_step_size: 30
scheduler_gamma: 0.5
```

### 1.2 MultiStepLR（多步长调度器）
**调度方式**：在指定的里程碑（epoch）处将学习率乘以衰减因子
**公式**：`lr = lr * gamma^k`，其中k是已通过的里程碑数量
**主要参数**：
- `milestones`：学习率衰减的里程碑列表（epoch数）
- `gamma`：衰减因子（默认0.1）
**适用场景**：训练过程中有明确的关键阶段需要调整学习率
**配置示例**：
```yaml
scheduler: "MultiStepLR"
scheduler_milestones: [50, 100, 150]
scheduler_gamma: 0.5
```

### 1.3 ExponentialLR（指数调度器）
**调度方式**：每个epoch将学习率乘以固定的衰减因子
**公式**：`lr = lr * gamma^epoch`
**主要参数**：
- `gamma`：每个epoch的衰减因子（默认0.95）
**适用场景**：需要平滑、连续的学习率衰减
**配置示例**：
```yaml
scheduler: "ExponentialLR"
scheduler_gamma: 0.95
```

## 2. 余弦调度器

### 2.1 CosineAnnealingLR（余弦退火调度器）
**调度方式**：使用余弦函数在指定周期内将学习率从初始值衰减到最小值
**公式**：`η_t = η_min + 0.5*(η_max - η_min)*(1 + cos(T_cur/T_max * π))`
**主要参数**：
- `T_max`：余弦周期的半周期长度（epoch数）
- `eta_min`：最小学习率（默认0）
**适用场景**：需要周期性重启的优化，避免陷入局部最小值
**配置示例**：
```yaml
scheduler: "CosineAnnealingLR"
min_lr: 1e-6
# T_max 自动计算为 num_epochs - warmup_epochs
```

### 2.2 CosineAnnealingWarmRestarts（余弦退火热重启调度器）
**调度方式**：余弦退火配合周期性热重启，每次重启后周期长度增加
**公式**：`η_t = η_min + 0.5*(η_max - η_min)*(1 + cos(T_cur/T_i * π))`
**主要参数**：
- `T_0`：第一次重启的周期长度
- `T_mult`：周期长度倍增因子（默认1）
- `eta_min`：最小学习率（默认0）
**适用场景**：需要周期性探索不同学习率区域的复杂优化问题
**配置示例**：
```yaml
scheduler: "CosineWarmRestarts"
min_lr: 1e-6
# 项目中默认使用 T_0=10, T_mult=2
```

## 3. 自适应调度器

### 3.1 ReduceLROnPlateau（高原衰减调度器）
**调度方式**：当监控指标停止改善时降低学习率
**触发条件**：连续`patience`个epoch指标没有改善超过`threshold`
**主要参数**：
- `mode`：监控模式，'min'或'max'
- `factor`：衰减因子（默认0.1）
- `patience`：等待改善的epoch数（默认10）
- `min_lr`：最小学习率（默认0）
- `threshold`：改善阈值（默认1e-4）
**适用场景**：验证损失或指标平台期时自动调整学习率
**配置示例**：
```yaml
scheduler: "ReduceLROnPlateau"
patience: 15
scheduler_factor: 0.5
min_lr: 1e-6
```

## 4. 循环调度器

### 4.1 CyclicLR（循环学习率调度器）
**调度方式**：在基础学习率和最大学习率之间循环变化
**策略模式**：
- `triangular`：线性上升和下降
- `triangular2`：每个周期振幅减半
- `exp_range`：指数变化
**主要参数**：
- `base_lr`：基础学习率
- `max_lr`：最大学习率
- `step_size_up`：上升步数
- `step_size_down`：下降步数
- `mode`：循环模式
**适用场景**：需要探索不同学习率范围的训练，有助于跳出局部最小值
**配置示例**：
```python
# YAML配置需要扩展支持
scheduler: "CyclicLR"
base_lr: 1e-5
max_lr: 1e-3
step_size_up: 2000
mode: "triangular"
```

### 4.2 OneCycleLR（单周期调度器）
**调度方式**：单个大周期内学习率从初始值上升到最大值再下降
**策略特点**：
- 学习率热身阶段
- 退火阶段
- 最终衰减阶段
**主要参数**：
- `max_lr`：最大学习率
- `total_steps`：总步数
- `pct_start`：上升阶段占比（默认0.3）
- `div_factor`：初始学习率 = max_lr/div_factor
- `final_div_factor`：最终学习率 = initial_lr/final_div_factor
**适用场景**：快速收敛的训练策略，常用于计算机视觉任务
**配置示例**：
```python
# YAML配置需要扩展支持
scheduler: "OneCycleLR"
max_lr: 1e-3
total_steps: 10000
pct_start: 0.3
div_factor: 25
final_div_factor: 1e4
```

## 5. 线性与多项式调度器

### 5.1 LinearLR（线性调度器）
**调度方式**：在指定迭代次数内线性调整学习率
**公式**：`lr = start_factor + (end_factor - start_factor) * (iter/total_iters)`
**主要参数**：
- `start_factor`：起始因子（默认1/3）
- `end_factor`：结束因子（默认1.0）
- `total_iters`：总迭代次数
**适用场景**：学习率热身或线性衰减
**配置示例**：
```yaml
scheduler: "LinearLR"
# 通常用于热身阶段
```

### 5.2 PolynomialLR（多项式调度器）
**调度方式**：使用多项式函数衰减学习率
**公式**：`lr = initial_lr * (1 - iter/total_iters)^power`
**主要参数**：
- `total_iters`：总迭代次数
- `power`：多项式幂（默认1.0，线性衰减）
**适用场景**：需要自定义衰减曲线的训练
**配置示例**：
```yaml
scheduler: "PolynomialLR"
total_iters: 10000
power: 0.9
```

## 6. 组合与自定义调度器

### 6.1 SequentialLR（顺序调度器）
**调度方式**：按顺序应用多个调度器，在指定里程碑切换
**主要参数**：
- `schedulers`：调度器列表
- `milestones`：切换点列表
**适用场景**：复杂的多阶段训练策略
**配置示例**：
```python
# 示例：先热身，再余弦退火
scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=5)
scheduler2 = CosineAnnealingLR(optimizer, T_max=195, eta_min=1e-6)
scheduler = SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[5])
```

### 6.2 ChainedScheduler（链式调度器）
**调度方式**：链式应用多个调度器，每个调度器在前一个的基础上调整
**主要参数**：
- `schedulers`：调度器列表
**适用场景**：需要叠加多种调整策略

### 6.3 LambdaLR（Lambda调度器）
**调度方式**：使用自定义lambda函数调整学习率
**公式**：`lr = initial_lr * lr_lambda(epoch)`
**主要参数**：
- `lr_lambda`：自定义函数或函数列表
**适用场景**：完全自定义的学习率调整策略
**配置示例**：
```python
# 自定义学习率调整
lambda1 = lambda epoch: 0.95 ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
```

### 6.4 MultiplicativeLR（乘法调度器）
**调度方式**：使用lambda函数乘法调整学习率
**公式**：`lr = previous_lr * lr_lambda(epoch)`
**主要参数**：
- `lr_lambda`：乘法因子函数
**适用场景**：相对调整而非绝对调整

## 7. 本项目支持的调度器

本项目目前支持以下调度器（通过`Module/Model/train.py`中的`_create_scheduler`方法）：

### 7.1 已实现的调度器
1. **ReduceLROnPlateau** - 高原衰减调度器
2. **CosineAnnealingLR** - 余弦退火调度器  
3. **CosineAnnealingWarmRestarts** - 余弦退火热重启调度器
4. **StepLR** - 步长调度器
5. **MultiStepLR** - 多步长调度器

### 7.2 热身调度器
项目还支持学习率热身（warmup）功能：
- **LambdaLR**：线性热身策略
- 配置参数：`warmup_epochs`
- 热身公式：`lr = initial_lr * (epoch + 1) / warmup_epochs`

### 7.3 配置示例
```yaml
training:
  scheduler: "CosineWarmRestarts"  # 或 "StepLR", "MultiStepLR", "Cosine", "ReduceLROnPlateau"
  patience: 15                     # ReduceLROnPlateau专用
  min_lr: 1e-6                     # 最小学习率
  scheduler_factor: 0.5            # 衰减因子
  scheduler_step_size: 30          # StepLR步长
  scheduler_gamma: 0.5             # 衰减系数
  scheduler_milestones: [50, 100, 150]  # MultiStepLR里程碑
  warmup_epochs: 5                 # 热身epoch数
```

## 8. 选择指南

### 8.1 根据任务类型选择
- **图像分类/检测**：CosineAnnealingLR, OneCycleLR
- **自然语言处理**：ReduceLROnPlateau, StepLR
- **生成模型**：CosineAnnealingWarmRestarts, CyclicLR
- **医学图像**：ReduceLROnPlateau, CosineAnnealingLR

### 8.2 根据训练阶段选择
- **初始训练**：使用热身 + CosineAnnealingLR
- **微调**：使用ReduceLROnPlateau
- **探索性训练**：使用CyclicLR或CosineAnnealingWarmRestarts

### 8.3 参数调优建议
1. **学习率范围**：初始学习率通常在1e-4到1e-3之间
2. **衰减因子**：gamma通常为0.1-0.5
3. **耐心值**：ReduceLROnPlateau的patience为总epoch数的5-10%
4. **最小学习率**：min_lr设为初始学习率的1/100到1/1000

## 9. 最佳实践

### 9.1 监控学习率
- 使用TensorBoard监控学习率变化
- 记录每个epoch的学习率值
- 分析学习率与损失曲线的相关性

### 9.2 调试技巧
1. **学习率过高**：损失NaN或爆炸 → 降低初始学习率
2. **学习率过低**：收敛缓慢 → 提高初始学习率或使用热身
3. **平台期**：验证损失停滞 → 使用ReduceLROnPlateau
4. **周期性波动**：使用余弦退火平滑变化

### 9.3 组合策略
- **热身+余弦退火**：先线性热身，再余弦退火
- **高原衰减+早停**：ReduceLROnPlateau配合早停机制
- **多阶段调度**：不同训练阶段使用不同调度器

## 10. 扩展支持

如需支持更多PyTorch调度器，可扩展`_create_scheduler`方法：

```python
# 添加新调度器支持示例
elif scheduler_name == "onecycle":
    return optim.lr_scheduler.OneCycleLR(
        self.optimizer,
        max_lr=self.config.training.learning_rate,
        total_steps=self.config.training.num_epochs * len(self.train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1e4
    )
```

通过合理选择和学习率调度器，可以显著提高模型训练效果和收敛速度。
