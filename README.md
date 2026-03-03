# 低剂量CT图像增强AI

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3+-green.svg)](https://monai.io/)
[![许可证: MIT](https://img.shields.io/badge/许可证-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于深度学习的低剂量CT图像增强框架，使用先进的神经网络架构和全面的诊断工具。

## 概述

低剂量CT图像增强AI是一个基于PyTorch的框架，旨在提高低剂量计算机断层扫描（CT）图像的质量。该项目利用最先进的深度学习模型来减少噪声并增强图像清晰度，使低剂量CT图像在质量上接近标准剂量CT图像。

该框架提供了完整的数据预处理、模型训练、推理和全面诊断的流程，适用于研究和临床应用。

## 主要特性

- **多种模型架构**：支持UNet2D/3D、Attention UNet、ResUNet、DenseUNet、WaveletDomainCNN和多尺度模型
- **完整的数据流程**：内置支持NIfTI、DICOM、PNG、JPG和NumPy格式，具有高级预处理功能
- **先进的训练框架**：混合精度训练、梯度裁剪、学习率调度和早停机制
- **诊断工具包**：全面的指标计算（PSNR、SSIM、RMSE、MAE、LPIPS）和模型诊断
- **可视化工具**：并排对比、训练曲线和诊断报告
- **模块化设计**：配置、数据加载、模型、训练和推理模块的清晰分离
- **生产就绪**：命令行界面、配置管理和检查点处理

## 安装

### 先决条件

- Python 3.9 或更高版本
- 建议使用CUDA兼容的GPU进行训练
- 3D模型训练需要8GB+ RAM

### 逐步安装

1. **克隆仓库**：
   ```bash
   git clone https://github.com/yourusername/low-dose-ct-enhancement.git
   cd low-dose-ct-enhancement
   ```

2. **创建虚拟环境**（推荐）：
   ```bash
   # 使用conda
   conda create -n ct-enhance python=3.9
   conda activate ct-enhance
   
   # 或使用venv
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

4. **验证安装**：
   ```bash
   python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
   python -c "import monai; print(f'MONAI版本: {monai.__version__}')"
   ```

## 快速开始

### 1. 创建示例数据

```bash
# 创建2D示例数据（20个样本）
python scripts/main.py create-data --type 2d --num 20

# 创建3D示例数据（5个样本）
python scripts/main.py create-data --type 3d --num 5
```

### 2. 训练模型

```bash
# 使用默认配置训练
python scripts/main.py train

# 使用自定义配置训练
python scripts/main.py train --config configs/advanced_training_config.yaml

# 从检查点恢复训练
python scripts/main.py train --resume ./checkpoints/checkpoint_epoch_20.pth

# 使用高级训练参数
python scripts/main.py train \
  --weight-decay 0.0001 \
  --gradient-clip 1.0 \
  --warmup-epochs 2 \
  --patience 10 \
  --loss-weights "1.0,0.5,0.1" \
  --multi-scale \
  --multi-scale-weights "1.0,0.5,0.25" \
  --experiment "my_experiment"

# 结合配置文件和命令行参数（命令行优先级更高）
python scripts/main.py train \
  --config configs/advanced_training_config.yaml \
  --resume ./custom_checkpoint.pth \
  --epochs 100 \
  --learning-rate 0.0005
```

### 3. 增强图像

```bash
# 增强单张图像
python scripts/main.py enhance --checkpoint models/checkpoints/best_model.pth --input data/low_dose/image.nii

# 增强目录中的所有图像
python scripts/main.py enhance --checkpoint models/checkpoints/best_model.pth --input data/low_dose/ --output results/
```

### 4. 运行诊断

```bash
# 为训练好的模型生成全面诊断报告
python -m Module.Tools.diagnostics comprehensive \
  --model models/checkpoints/best_model.pth \
  --data ./data \
  --log ./training_logs/training_log.json \
  --output diagnostics/comprehensive
```

## 项目结构

```
.
├── .gitignore
├── LICENSE
├── README.md
├── requirements-dev.txt
├── requirements.txt
├── run_tests.py
├── tox.ini
├── .github/
├── .pytest_cache/
├── configs/
│   └── advanced_training_config.yaml
├── diagnostics/
│   ├── reports/
│   │   └── optimization_validation_20260222_164547.json
│   └── visualizations/
├── Module/
│   ├── __init__.py
│   ├── Config/
│   │   └── config.py
│   ├── Inference/
│   │   └── inference.py
│   ├── Loader/
│   │   ├── data_loader.py
│   │   └── optimized_data_loader.py
│   ├── Model/
│   │   ├── losses.py
│   │   ├── models.py
│   │   └── train.py
│   └── Tools/
│       ├── amp_optimizer.py
│       ├── device_manager.py
│       ├── diagnostics.py
│       ├── memory_optimizer.py
│       ├── performance_monitor.py
│       ├── utils.py
│       ├── wavelet_transform.py
│       └── diagnostics/
│           ├── __init__.py
│           ├── optimized_diagnostics.py
│           ├── analysis/
│           │   └── __init__.py
│           ├── cli/
│           │   └── __init__.py
│           ├── config/
│           │   └── __init__.py
│           ├── metrics/
│           ├── model/
│           │   └── __init__.py
│           ├── utils/
│           │   └── optimization.py
│           └── visualization/
│               └── __init__.py
├── scripts/
│   ├── main.py
│   ├── README.md
│   ├── validate_optimizations.py
│   ├── inference/
│   ├── training/
│   │   └── train_advanced.py
│   └── utils/
│       ├── arg_parser.py
│       ├── config_loader.py
│       ├── error_handler.py
│       └── logging.py
```

## 配置

项目使用分层配置系统，包含四个主要组件：

### 1. 数据配置 (`DataConfig`)
- `data_dir`: 数据根目录
- `low_dose_dir`: 低剂量CT图像子目录
- `full_dose_dir`: 全剂量CT图像子目录（目标）
- `image_size`: 图像尺寸（高、宽、深）
- `batch_size`: 训练批大小
- `normalize_range`: CT值归一化范围（默认：[-1000, 1000] HU）

### 2. 模型配置 (`ModelConfig`)
- `model_name`: 架构名称（UNet2D、AttentionUNet、WaveletDomainCNN等）
- `in_channels`: 输入通道数（灰度CT为1）
- `out_channels`: 输出通道数（增强CT为1）
- `features`: 每层的特征通道数
- `dropout`: 正则化丢弃率

### 3. 训练配置 (`TrainingConfig`)
- `learning_rate`: 初始学习率
- `num_epochs`: 总训练轮数
- `loss_function`: 损失函数（L1Loss、MSELoss、SSIMLoss、MixedLoss、MultiScaleLoss）
- `optimizer`: 优化器（Adam、AdamW、SGD）
- `scheduler`: 学习率调度器（ReduceLROnPlateau、Cosine、Step、MultiStep、CosineWarmRestarts）
- `use_amp`: 启用自动混合精度训练
- `weight_decay`: 权重衰减系数
- `patience`: 早停耐心值
- `warmup_epochs`: 学习率热身轮数
- `gradient_clip_value`: 梯度裁剪值
- `loss_weights`: 混合损失权重（用于MixedLoss）
- `use_multi_scale_loss`: 是否使用多尺度损失
- `multi_scale_weights`: 多尺度损失权重
- `resume_checkpoint`: 从检查点恢复训练的路径（支持命令行和配置文件）

### 4. 诊断配置 (`DiagnosticsConfig`)
- `enable_diagnostics`: 启用诊断功能
- `compute_psnr/ssim/rmse/mae`: 启用特定指标
- `visualize_samples`: 可视化样本数量
- `generate_html_report`: 生成HTML诊断报告

### 配置示例

创建YAML配置文件：

```yaml
data:
  data_dir: "./data"
  low_dose_dir: "qd"
  full_dose_dir: "fd"
  image_size: [512, 512, 1]
  batch_size: 4

model:
  model_name: "WaveletDomainCNN"
  features: [32, 64, 128, 256]
  dropout: 0.1

training:
  learning_rate: 1e-4
  num_epochs: 200
  loss_function: "MixedLoss"
  optimizer: "AdamW"
  scheduler: "CosineWarmRestarts"
  weight_decay: 1e-5
  patience: 15
  warmup_epochs: 0
  gradient_clip_value: 1.0
  loss_weights: [1.0, 0.5, 0.1]
  use_multi_scale_loss: true
  multi_scale_weights: [1.0, 0.5, 0.25]
  # 可选：从检查点恢复训练
  # resume_checkpoint: "./checkpoints/checkpoint_epoch_20.pth"
```

## 学习率调度器详细说明

学习率调度器是深度学习训练中的关键组件，用于在训练过程中动态调整学习率，以改善模型收敛性和性能。PyTorch提供了多种学习率调度器，本项目支持其中主要的几种。

### 1. 本项目支持的调度器

#### 1.1 ReduceLROnPlateau（高原衰减调度器）
**调度方式**：当验证指标停止改善时降低学习率
**主要参数**：
- `patience`：等待改善的epoch数（默认15）
- `factor`：衰减因子（默认0.5）
- `min_lr`：最小学习率（默认1e-6）
**适用场景**：验证损失或指标平台期时自动调整学习率
**配置示例**：
```yaml
scheduler: "ReduceLROnPlateau"
patience: 15
scheduler_factor: 0.5
min_lr: 1e-6
```

#### 1.2 CosineAnnealingLR（余弦退火调度器）
**调度方式**：使用余弦函数在指定周期内将学习率从初始值衰减到最小值
**主要参数**：
- `T_max`：余弦周期的半周期长度（自动计算为`num_epochs - warmup_epochs`）
- `eta_min`：最小学习率（默认1e-6）
**适用场景**：需要平滑衰减的学习率策略
**配置示例**：
```yaml
scheduler: "Cosine"
min_lr: 1e-6
```

#### 1.3 CosineAnnealingWarmRestarts（余弦退火热重启调度器）
**调度方式**：余弦退火配合周期性热重启，每次重启后周期长度增加
**主要参数**：
- `T_0`：第一次重启的周期长度（默认10）
- `T_mult`：周期长度倍增因子（默认2）
- `eta_min`：最小学习率（默认1e-6）
**适用场景**：需要周期性探索不同学习率区域的复杂优化问题
**配置示例**：
```yaml
scheduler: "CosineWarmRestarts"
min_lr: 1e-6
```

#### 1.4 StepLR（步长调度器）
**调度方式**：每隔固定步数将学习率乘以衰减因子
**主要参数**：
- `step_size`：学习率衰减的步长（默认30）
- `gamma`：衰减因子（默认0.5）
**适用场景**：简单的周期性学习率衰减
**配置示例**：
```yaml
scheduler: "StepLR"
scheduler_step_size: 30
scheduler_gamma: 0.5
```

#### 1.5 MultiStepLR（多步长调度器）
**调度方式**：在指定的里程碑处将学习率乘以衰减因子
**主要参数**：
- `milestones`：学习率衰减的里程碑列表（默认[50, 100, 150]）
- `gamma`：衰减因子（默认0.5）
**适用场景**：训练过程中有明确的关键阶段需要调整学习率
**配置示例**：
```yaml
scheduler: "MultiStepLR"
scheduler_milestones: [50, 100, 150]
scheduler_gamma: 0.5
```

### 2. 热身调度器（Warmup）

项目支持学习率热身功能，在训练初期逐步提高学习率：
- **配置参数**：`warmup_epochs`
- **热身公式**：`lr = initial_lr * (epoch + 1) / warmup_epochs`
- **配置示例**：
```yaml
warmup_epochs: 5
```

### 3. 其他PyTorch调度器

PyTorch还支持以下调度器，如需使用可扩展项目代码：

#### 3.1 ExponentialLR（指数调度器）
- **调度方式**：每个epoch将学习率乘以固定的衰减因子
- **公式**：`lr = lr * gamma^epoch`
- **适用场景**：需要平滑、连续的学习率衰减

#### 3.2 CyclicLR（循环学习率调度器）
- **调度方式**：在基础学习率和最大学习率之间循环变化
- **策略模式**：triangular（线性）、triangular2（振幅减半）、exp_range（指数）
- **适用场景**：需要探索不同学习率范围的训练

#### 3.3 OneCycleLR（单周期调度器）
- **调度方式**：单个大周期内学习率从初始值上升到最大值再下降
- **适用场景**：快速收敛的训练策略，常用于计算机视觉任务

#### 3.4 LinearLR（线性调度器）
- **调度方式**：在指定迭代次数内线性调整学习率
- **适用场景**：学习率热身或线性衰减

#### 3.5 PolynomialLR（多项式调度器）
- **调度方式**：使用多项式函数衰减学习率
- **公式**：`lr = initial_lr * (1 - iter/total_iters)^power`
- **适用场景**：需要自定义衰减曲线的训练

### 4. 组合调度器

#### 4.1 SequentialLR（顺序调度器）
- **调度方式**：按顺序应用多个调度器，在指定里程碑切换
- **适用场景**：复杂的多阶段训练策略

#### 4.2 ChainedScheduler（链式调度器）
- **调度方式**：链式应用多个调度器，每个调度器在前一个的基础上调整

#### 4.3 LambdaLR（Lambda调度器）
- **调度方式**：使用自定义lambda函数调整学习率
- **适用场景**：完全自定义的学习率调整策略

### 5. 选择指南

#### 5.1 根据任务类型选择
- **图像分类/检测**：CosineAnnealingLR, OneCycleLR
- **自然语言处理**：ReduceLROnPlateau, StepLR
- **生成模型**：CosineAnnealingWarmRestarts, CyclicLR
- **医学图像**：ReduceLROnPlateau, CosineAnnealingLR

#### 5.2 根据训练阶段选择
- **初始训练**：使用热身 + CosineAnnealingLR
- **微调**：使用ReduceLROnPlateau
- **探索性训练**：使用CyclicLR或CosineAnnealingWarmRestarts

#### 5.3 参数调优建议
1. **学习率范围**：初始学习率通常在1e-4到1e-3之间
2. **衰减因子**：gamma通常为0.1-0.5
3. **耐心值**：ReduceLROnPlateau的patience为总epoch数的5-10%
4. **最小学习率**：min_lr设为初始学习率的1/100到1/1000

### 6. 最佳实践

#### 6.1 监控学习率
- 使用TensorBoard监控学习率变化：`tensorboard --logdir logs/`
- 记录每个epoch的学习率值
- 分析学习率与损失曲线的相关性

#### 6.2 调试技巧
1. **学习率过高**：损失NaN或爆炸 → 降低初始学习率
2. **学习率过低**：收敛缓慢 → 提高初始学习率或使用热身
3. **平台期**：验证损失停滞 → 使用ReduceLROnPlateau
4. **周期性波动**：使用余弦退火平滑变化

#### 6.3 配置示例组合
```yaml
# 示例1：热身+余弦退火
warmup_epochs: 5
scheduler: "Cosine"
min_lr: 1e-6

# 示例2：高原衰减+早停
scheduler: "ReduceLROnPlateau"
patience: 15
scheduler_factor: 0.5
min_lr: 1e-6
use_early_stopping: true
early_stopping_patience: 30

# 示例3：多阶段调度
scheduler: "MultiStepLR"
scheduler_milestones: [50, 100, 150]
scheduler_gamma: 0.5
```

## 使用示例

### Python API 使用

```python
from Module.Config.config import Config
from Module.Model.train import Trainer
from Module.Inference.inference import CTEnhancer

# 创建自定义配置
config = Config()
config.data.batch_size = 8
config.training.num_epochs = 150
config.model.model_name = "AttentionUNet"

# 训练模型
trainer = Trainer(config)
trainer.train()

# 从检查点恢复训练
trainer_resume = Trainer(config, checkpoint_path="./checkpoints/checkpoint_epoch_20.pth")
trainer_resume.train()

# 使用高级训练参数
config.training.weight_decay = 0.0001
config.training.patience = 10
setattr(config.training, 'gradient_clip_value', 1.0)
setattr(config.training, 'warmup_epochs', 2)
config.training.loss_weights = (1.0, 0.5, 0.1)
config.training.use_multi_scale_loss = True
config.training.multi_scale_weights = (1.0, 0.5, 0.25)

trainer_advanced = Trainer(config)
trainer_advanced.train()

# 增强图像
enhancer = CTEnhancer("models/checkpoints/best_model.pth")
enhanced_image = enhancer.enhance(low_dose_image)
enhancer.enhance_file("input.nii", "output.nii")
```

### 命令行使用

```bash
# 使用特定配置训练
python scripts/main.py train --config my_config.yaml

# 使用模式匹配增强多个文件
python scripts/main.py enhance \
  --checkpoint models/checkpoints/best_model.pth \
  --input data/low_dose/ \
  --output results/ \
  --pattern "*.nii.gz"

# 创建3D虚拟数据用于测试
python scripts/main.py create-data --type 3d --num 10
```

## 诊断工具

框架包含全面的诊断工具，用于模型评估和故障排除：

### 可用诊断功能

1. **图像质量指标**：
   - PSNR（峰值信噪比）
   - SSIM（结构相似性指数）
   - MS-SSIM（多尺度SSIM）
   - RMSE（均方根误差）
   - MAE（平均绝对误差）
   - LPIPS（学习感知图像块相似度）

2. **模型诊断**：
   - 梯度流分析
   - 权重分布监控
   - 激活统计
   - 死亡ReLU检测

3. **训练分析**：
   - 过拟合检测
   - 学习率监控
   - 损失曲线分析
   - 收敛诊断

### 使用诊断工具

```python
from Module.Tools.diagnostics import DiagnosticsCLI
from Module.Tools.diagnostics import DiagnosticsConfig

# 创建诊断配置和CLI
config = DiagnosticsConfig(
    compute_psnr=True,
    compute_ssim=True,
    visualize_samples=3,
    check_gradients=True
)
cli = DiagnosticsCLI(config)

# 运行全面诊断
report = cli.run_comprehensive_diagnostics(
    model_path="models/checkpoints/best_model.pth",
    data_path="./data",
    training_log_path="./training_logs/training_log.json",
    output_dir="./diagnostics/comprehensive"
)

# 生成HTML报告
cli.generate_html_report(report, "diagnostics/report.html")
```

### 命令行诊断

```bash
# 运行全面诊断
python -m Module.Tools.diagnostics comprehensive \
  --model models/checkpoints/best_model.pth \
  --data ./data \
  --log ./training_logs/training_log.json \
  --output diagnostics/comprehensive

# 运行指标分析
python -m Module.Tools.diagnostics metrics \
  --pred ./data/predictions \
  --target ./data/targets \
  --output diagnostics/metrics

# 运行模型诊断
python -m Module.Tools.diagnostics model \
  --model models/checkpoints/best_model.pth \
  --data ./data/samples \
  --output diagnostics/model

# 运行训练分析
python -m Module.Tools.diagnostics training \
  --log ./training_logs/training_log.json \
  --output diagnostics/training
```

## 支持的模型

### 1. **UNet2D/3D**
标准的U-Net架构，用于2D/3D图像增强，具有编码器-解码器结构和跳跃连接。

### 2. **Attention UNet**
带有注意力门的U-Net，专注于相关结构同时抑制无关区域。

### 3. **ResUNet**
带有残差连接的U-Net，改善梯度流和更深网络训练。

### 4. **DenseUNet**
密集连接的U-Net，跨多层重用特征。

### 5. **WaveletDomainCNN**
基于小波变换的CNN，在频域处理图像以进行噪声减少。

### 6. **MultiScaleModel**
多尺度架构，在不同分辨率处理图像以捕获局部和全局特征。

### 7. **FBPConvNet**
滤波反投影卷积网络，专门设计用于CT重建任务。

## 数据格式支持

- **NIfTI** (.nii, .nii.gz): 标准医学图像格式
- **DICOM** (.dcm): 医学影像标准（需要pydicom）
- **图像格式** (.png, .jpg, .bmp): 2D图像
- **NumPy数组** (.npy): 原始数组数据
- **MATLAB文件** (.mat): MATLAB数据文件

## 性能优化

### GPU加速
- 启用CUDA进行GPU加速
- 使用混合精度训练，设置 `use_amp: true`
- 实现梯度检查点以提高内存效率

### 数据加载优化
- 使用MONAI的缓存数据集
- 使用 `num_workers` 实现多线程数据加载
- 使用持久化工作器减少开销

### 训练优化
- 梯度累积以获得更大的有效批大小
- 自动混合精度训练
- 分布式数据并行用于多GPU训练

## 故障排除

### 常见问题

1. **内存不足（OOM）错误**：
   - 减少批大小
   - 使用梯度累积
   - 启用混合精度训练
   - 使用较小的图像尺寸

2. **数据加载缓慢**：
   - 增加 `num_workers`（Windows上设置为0）
   - 使用MONAI的缓存或持久化数据集
   - 离线预处理数据

3. **模型收敛不佳**：
   - 调整学习率
   - 尝试不同的损失函数
   - 检查数据归一化
   - 验证标签对齐

4. **文件格式问题**：
   - 确保安装了必需的库（nibabel、pydicom）
   - 检查文件权限和路径
   - 验证文件完整性

### 日志和监控

- TensorBoard日志：`tensorboard --logdir logs/`
- 控制台输出损失和指标
- 检查点文件在 `models/checkpoints/`
- 诊断报告在 `diagnostics/`

## 贡献指南

我们欢迎贡献来改进低剂量CT图像增强AI框架！

### 开发环境设置

1. Fork仓库
2. 克隆你的fork：
   ```bash
   git clone https://github.com/yourusername/low-dose-ct-enhancement.git
   cd low-dose-ct-enhancement
   ```

3. 安装开发依赖：
   ```bash
   pip install -r requirements.txt
   pip install black flake8 mypy pytest
   ```

4. 创建功能分支：
   ```bash
   git checkout -b feature/你的功能名称
   ```

### 代码风格

- 遵循PEP 8指南
- 函数签名使用类型提示
- 为所有公共函数和类编写文档字符串
- 保持函数专注和模块化

### 测试

- 为新功能编写单元测试
- 确保现有测试通过：
  ```bash
  python -m pytest tests/
  ```
- 使用不同配置和数据类型进行测试

### Pull Request流程

1. 确保你的代码通过所有测试
2. 如果需要，更新文档
3. 添加描述性的提交信息
4. 创建具有清晰描述的pull request

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 引用

如果在研究中使用此框架，请引用：

```bibtex
@software{low_dose_ct_enhancement_2025,
  title = {低剂量CT图像增强AI},
  author = {Joker_1212},
  year = {2025},
  url = {https://github.com/yourusername/low-dose-ct-enhancement},
  note = {用于增强低剂量CT图像的深度学习框架}
}
```

## 参考文献

1. MONAI: Medical Open Network for AI - https://monai.io/
2. "Low-Dose CT Image Denoising Using a Generative Adversarial Network" - IEEE Transactions on Medical Imaging
3. "Attention U-Net: Learning Where to Look for the Pancreas" - MIDL 2018
4. "Wavelet Domain Deep Learning for Medical Image Analysis" - Medical Image Analysis

## 致谢

- MONAI团队提供的优秀医学成像框架
- PyTorch社区的深度学习生态系统
- 提供反馈和改进的贡献者和用户

## 联系方式

如有问题、问题或贡献：
- 在 [GitHub](https://github.com/yourusername/low-dose-ct-enhancement/issues) 上提交问题
- 电子邮件：your.email@example.com

---

**注意**：这是一个研究框架。临床使用需要适当的验证和监管批准。
