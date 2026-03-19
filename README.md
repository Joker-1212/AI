# 低剂量CT图像增强AI

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-red.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.5+-green.svg)](https://monai.io/)
[![许可证: MIT](https://img.shields.io/badge/许可证-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于深度学习的低剂量CT图像增强框架。核心模型在轮廓波/小波变换域中使用 U-Net 进行去噪，结合 Refinement U-Net 在空间域精修，使低剂量CT图像接近全剂量质量。

## 概述

本框架基于 PyTorch 和 MONAI 构建，提供从数据预处理、模型训练到推理的完整流程。默认模型 `WaveletDomainCNN` 采用以下架构：

```
输入 → 轮廓波变换（固定拉普拉斯金字塔 + 固定方向滤波器组）
     → 共享 U-Net 处理各层方向细节子带（低频直接传递）
     → 轮廓波逆变换重构
     → Refinement U-Net 空间域精修
     → 输出
```

备选 DWT 模式：`输入 → DWT(纯PyTorch) → U-Net处理子带 → 逆DWT → Refinement U-Net → 输出`

## 主要特性

- **变换域去噪**：轮廓波变换（默认）或 DWT 将图像分解到频域，U-Net 在子带上去噪，逆变换重构
- **非可学习变换**：方向滤波器组和小波滤波器均为固定参数（`register_buffer`），不参与梯度更新
- **多种模型架构**：UNet2D/3D、Attention UNet、ResUNet、DenseUNet、WaveletDomainCNN、多尺度模型、FBPConvNet
- **完整数据流程**：支持 NIfTI、DICOM、PNG、NumPy 格式，自动 80/10/10 划分
- **训练框架**：混合精度(AMP)、梯度裁剪/累积、6种学习率调度器、Warmup、早停、检查点恢复
- **诊断工具**：PSNR、SSIM、LPIPS 等指标计算，梯度流分析，训练曲线可视化
- **跨平台**：自动检测 CUDA/MPS/CPU，Windows 优化数据加载器

## 安装

### 环境要求

- Python 3.13+
- 建议使用 CUDA 兼容 GPU 进行训练
- 8GB+ 显存（两个 U-Net 参数量较大）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/yourusername/low-dose-ct-enhancement.git
cd low-dose-ct-enhancement

# 创建环境
conda create -n ct-enhance python=3.13
conda activate ct-enhance

# 安装依赖
pip install -r requirements.txt

# 验证
python -c "import torch; print(torch.__version__)"
python -c "import monai; print(monai.__version__)"
```

## 快速开始

### 创建示例数据

```bash
python scripts/main.py create-data --type 2d --num 20
python scripts/main.py create-data --type 3d --num 5
```

### 训练模型

```bash
# 默认配置（轮廓波 + U-Net）
python scripts/main.py train

# 使用自定义配置
python scripts/main.py train --config configs/advanced_training_config.yaml

# 从检查点恢复训练
python scripts/main.py train --resume ./checkpoints/checkpoint_epoch_20.pth
```

### 增强图像

```bash
python scripts/main.py enhance \
  --checkpoint models/checkpoints/best_model.pth \
  --input data/low_dose/image.nii
```

### 运行诊断

```bash
python -m Module.Tools.diagnostics comprehensive \
  --model models/checkpoints/best_model.pth \
  --data ./data
```

## 项目结构

```
.
├── configs/                          # YAML 配置文件
│   ├── advanced_training_config.yaml
│   ├── fast_config.yaml
│   └── simple_optimized_config.yaml
├── Module/
│   ├── Config/config.py              # 数据/模型/训练/诊断配置(dataclass)
│   ├── Loader/data_loader.py         # NIfTI/DICOM/PNG/NumPy 数据加载
│   ├── Model/
│   │   ├── models.py                 # 8种模型架构
│   │   ├── losses.py                 # L1/MSE/SSIM/Mixed/MultiScale/Perceptual 损失
│   │   └── train.py                  # Trainer 类
│   ├── Inference/inference.py        # CTEnhancer 推理类
│   └── Tools/
│       ├── wavelet_transform.py      # DWT2d / FixedDirectionalFilterBank / ContourletTransform
│       ├── device_manager.py         # CUDA/MPS/CPU 自动检测
│       ├── amp_optimizer.py          # 混合精度训练
│       └── diagnostics/              # 诊断子包(指标/可视化/分析)
├── scripts/main.py                   # CLI 入口
├── requirements.txt
└── README.md
```

## 配置

所有超参数通过 YAML 配置文件或 dataclass 默认值控制。支持环境变量覆盖。

### 配置示例

```yaml
data:
  data_dir: "./data"
  low_dose_dir: "qd"       # 低剂量子目录
  full_dose_dir: "fd"      # 全剂量子目录
  image_size: [512, 512, 1]
  batch_size: 8
  normalize_range: [-0.1, 0.1]

model:
  model_name: "WaveletDomainCNN"
  features: [32, 64, 128, 256]
  dropout: 0.1
  wavelet_type: "contourlet"   # 可选: "contourlet"(默认), "dwt"

training:
  learning_rate: 1e-4
  num_epochs: 200
  optimizer: "AdamW"
  loss_function: "MixedLoss"
  loss_weights: [1.0, 0.5, 0.1]    # [L1, SSIM, Perceptual]
  use_multi_scale_loss: false       # 轮廓波自带多尺度，默认关闭
  scheduler: "CosineWarmRestarts"
  use_amp: true
  gradient_clip_value: 1.0
```

### 小波类型选择

| 类型 | 说明 | 输出子带 | 适用场景 |
|------|------|----------|----------|
| `contourlet` | 拉普拉斯金字塔 + 固定方向滤波器组（8方向） | 多层 detail + low_freq | 默认，方向性特征丰富 |
| `dwt` | 纯 PyTorch 可分离 DWT（db4） | LL/LH/HL/HH 4子带 | 完美重构，计算更快 |

## 支持的模型

| 模型 | 说明 |
|------|------|
| **WaveletDomainCNN** | 变换域 U-Net 去噪 + Refinement U-Net（默认） |
| **UNet2D / UNet3D** | 标准 U-Net 编码器-解码器 |
| **AttentionUNet** | 注意力门 U-Net |
| **ResUNet** | 残差连接 U-Net |
| **DenseUNet** | 密集连接 U-Net |
| **MultiScaleModel** | 多尺度（原始/1:2/1:4）特征融合 |
| **FBPConvNet** | FBP + CNN 去噪 |

## 学习率调度器

| 调度器 | 关键参数 | 说明 |
|--------|----------|------|
| `ReduceLROnPlateau` | `patience`, `scheduler_factor` | 验证指标停滞时衰减 |
| `Cosine` | `min_lr` | 余弦退火 |
| `CosineWarmRestarts` | `min_lr` | 余弦退火 + 周期热重启 |
| `StepLR` | `scheduler_step_size`, `scheduler_gamma` | 固定步长衰减 |
| `MultiStepLR` | `scheduler_milestones`, `scheduler_gamma` | 指定里程碑衰减 |
| `ExponentialLR` | `scheduler_gamma` | 指数衰减 |

支持 `warmup_epochs` 热身（线性增长到初始学习率）。

## 损失函数

| 损失 | 说明 |
|------|------|
| `L1Loss` | 像素级 L1 |
| `MSELoss` | 像素级 MSE |
| `SSIMLoss` | 结构相似性 |
| `MixedLoss` | 加权组合 L1 + SSIM + Perceptual |
| `MultiScaleLoss` | 多尺度加权损失（DWT模式可选用） |
| `PerceptualLoss` | 感知损失 |

## 数据格式

- **NIfTI** (.nii, .nii.gz)
- **DICOM** (.dcm)
- **图像** (.png, .jpg, .bmp)
- **NumPy** (.npy)

数据目录约定：`data_dir/qd/` 放低剂量图像，`data_dir/fd/` 放全剂量图像。归一化范围默认 `[-0.1, 0.1]`。

## Python API

```python
from Module.Config.config import Config
from Module.Model.train import Trainer
from Module.Inference.inference import CTEnhancer

# 配置
config = Config()
config.model.model_name = "WaveletDomainCNN"
config.model.wavelet_type = "contourlet"  # 或 "dwt"
config.data.batch_size = 8
config.training.num_epochs = 200

# 训练
trainer = Trainer(config)
trainer.train()

# 推理
enhancer = CTEnhancer("models/checkpoints/best_model.pth")
enhanced = enhancer.enhance(low_dose_image)
enhancer.enhance_file("input.nii", "output.nii")
```

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| GPU OOM | 减小 `batch_size`，启用 `use_amp: true` |
| Windows 数据加载慢 | 设置 `num_workers: 0` |
| 收敛不佳 | 调整学习率，尝试不同 `loss_function`，检查归一化范围 |
| 旧检查点无法加载 | 架构变更后需重新训练 |

## 依赖

核心：PyTorch 2.9+, MONAI 1.5+, nibabel, pydicom, PyWavelets, NumPy, SciPy, matplotlib, PyYAML

完整列表见 `requirements.txt`，开发工具见 `requirements-dev.txt`。

## 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE)。

## 引用

```bibtex
@software{low_dose_ct_enhancement_2025,
  title = {低剂量CT图像增强AI},
  author = {Joker_1212},
  year = {2025},
  url = {https://github.com/yourusername/low-dose-ct-enhancement}
}
```

## 参考文献

1. MONAI: Medical Open Network for AI - https://monai.io/
2. "A deep convolutional neural network using directional wavelets for low-dose X-ray CT" - Medical Physics
3. "Attention U-Net: Learning Where to Look for the Pancreas" - MIDL 2018
4. "Low-Dose CT Image Denoising Using a Generative Adversarial Network" - IEEE TMI

---

**注意**：本框架用于研究目的。临床使用需要适当的验证和监管批准。
