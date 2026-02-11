# 低剂量CT图像增强AI

基于MONAI框架的深度学习模型，用于提高低剂量CT重建图像的清晰度。

## 项目概述

本项目旨在使用深度学习技术增强低剂量CT图像的质量，减少噪声并提高图像清晰度，使其接近标准剂量CT图像的质量。

### 主要功能

- **数据预处理**: 支持NIfTI、DICOM、PNG等多种格式的CT图像
- **多种模型架构**: UNet3D、Attention UNet、ResUNet、DenseUNet、多尺度模型
- **完整训练流程**: 数据增强、训练、验证、测试一体化
- **推理工具**: 对新的低剂量CT图像进行增强
- **可视化**: 结果对比和指标评估

## 项目结构

```
.
├── README.md                 # 项目说明
├── requirements.txt          # Python依赖
├── run.py                    # 主入口点脚本
├── src/                      # 源代码
│   ├── config.py            # 配置类
│   ├── data_loader.py       # 数据加载和预处理
│   ├── models.py            # 模型定义
│   ├── train.py             # 训练脚本
│   ├── inference.py         # 推理脚本
│   └── utils.py             # 工具函数
├── scripts/                  # 实用脚本
│   └── create_sample_data.py # 创建示例数据
├── data/                     # 数据目录（自动创建）
│   ├── low_dose/            # 低剂量CT图像
│   └── full_dose/           # 全剂量CT图像（目标）
├── configs/                  # 配置文件
├── models/                   # 训练好的模型
├── logs/                     # 训练日志（TensorBoard）
└── notebooks/               # Jupyter笔记本
```

## 快速开始

### 1. 安装依赖

```bash
# 使用conda环境（推荐）
conda create -n monai-ct python=3.9
conda activate monai-ct

# 安装依赖
pip install -r requirements.txt
```

### 2. 创建示例数据

```bash
# 创建2D示例数据
python run.py create-data --type 2d --num 20

# 或创建3D示例数据
python run.py create-data --type 3d --num 5
```

### 3. 训练模型

```bash
# 使用默认配置训练
python run.py train

# 使用自定义配置文件
python run.py train --config configs/custom_config.yaml
```

### 4. 增强图像

```bash
# 增强单张图像
python run.py enhance --checkpoint models/checkpoints/best_model.pth --input data/low_dose/image.nii

# 增强整个目录
python run.py enhance --checkpoint models/checkpoints/best_model.pth --input data/low_dose/ --output results/
```

## 配置说明

项目使用数据类进行配置管理，主要配置包括：

- **DataConfig**: 数据相关配置（路径、尺寸、批大小等）
- **ModelConfig**: 模型架构配置（模型类型、通道数、特征等）
- **TrainingConfig**: 训练参数（学习率、迭代次数、损失函数等）

可以通过修改 `src/config.py` 或创建YAML配置文件来自定义配置。

## 支持的模型

1. **UNet3D**: 标准的3D UNet架构
2. **Attention UNet**: 带注意力机制的UNet
3. **ResUNet**: 残差连接的UNet
4. **DenseUNet**: 密集连接UNet
5. **MultiScale**: 多尺度特征融合模型

## 数据格式支持

- **NIfTI (.nii, .nii.gz)**: 医学图像标准格式
- **DICOM (.dcm)**: 医疗影像格式（需要pydicom）
- **图像格式 (.png, .jpg, .bmp)**: 2D图像
- **NumPy数组 (.npy)**: 原始数据

## 评估指标

- **PSNR (峰值信噪比)**: 衡量图像质量
- **SSIM (结构相似性)**: 衡量结构相似度
- **L1/L2损失**: 像素级误差

## 使用示例

### 在Python中直接使用

```python
from src.inference import CTEnhancer

# 初始化增强器
enhancer = CTEnhancer("models/checkpoints/best_model.pth")

# 增强图像
enhanced_image = enhancer.enhance(low_dose_image)

# 或增强文件
enhancer.enhance_file("input.nii", "output.nii")
```

### 自定义训练

```python
from src.config import Config
from src.train import Trainer

# 自定义配置
config = Config()
config.data.batch_size = 8
config.training.num_epochs = 200
config.model.model_name = "AttentionUNet"

# 训练
trainer = Trainer(config)
trainer.train()
```

## 性能优化建议

1. **GPU加速**: 确保使用CUDA兼容的GPU
2. **数据预处理**: 使用MONAI的缓存数据集加速数据加载
3. **混合精度训练**: 使用`torch.cuda.amp`减少内存占用
4. **分布式训练**: 支持多GPU训练

## 故障排除

### 常见问题

1. **内存不足**: 减小批大小或图像尺寸
2. **数据加载慢**: 使用`num_workers`参数并行加载
3. **模型不收敛**: 调整学习率或尝试不同的损失函数
4. **文件格式不支持**: 确保安装了相应的库（如pydicom）

### 日志和监控

- 使用TensorBoard监控训练过程：`tensorboard --logdir logs/`
- 检查控制台输出中的损失和指标

## 参考文献

1. MONAI官方文档: https://docs.monai.io/
2. "Low-Dose CT Image Denoising Using a Generative Adversarial Network" - IEEE TMI
3. "Attention U-Net: Learning Where to Look for the Pancreas" - MIDL 2018

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进本项目。
