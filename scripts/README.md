# CT增强AI - 脚本文档

本目录包含低剂量CT增强AI项目的组织化脚本。

## 目录结构

```
scripts/
├── main.py                 # 主入口点，支持子命令
├── data/                   # 数据相关脚本
│   └── create_sample_data.py
├── training/               # 训练脚本
│   ├── train_advanced.py   # 高级训练脚本（支持超参数调整）
│   └── (未来: train_basic.py)
├── inference/              # 推理脚本
│   └── (未来: enhance_image.py, enhance_directory.py)
├── utils/                  # 工具模块
│   ├── logging.py         # 日志配置
│   ├── config_loader.py   # 配置管理
│   ├── error_handler.py   # 错误处理工具
│   └── arg_parser.py      # 命令行参数解析
└── README.md              # 本文档
```

## 快速开始

### 1. 主入口点

主要接口是 `main.py`，提供子命令功能：

```bash
# 显示帮助
python scripts/main.py --help

# 训练模型
python scripts/main.py train --config configs/advanced_training_config.yaml

# 增强单张图像
python scripts/main.py enhance --checkpoint checkpoints/best_model.pth --input data/test.nii --output enhanced.nii

# 增强目录中的图像
python scripts/main.py enhance --checkpoint checkpoints/best_model.pth --input data/test_dir/ --output enhanced_dir/

# 创建示例数据
python scripts/main.py create-data --type 2d --num 50

# 测试管道
python scripts/main.py test
```

### 2. 高级训练

如需更多训练参数控制，使用高级训练脚本：

```bash
python scripts/training/train_advanced.py \
  --epochs 100 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --model UNet2D \
  --data-dir ./data \
  --experiment my_experiment
```

### 3. 数据创建

创建用于测试的合成CT数据：

```bash
python scripts/data/create_sample_data.py --type 3d --num 10
```

## 脚本详情

### main.py

主脚本提供统一接口，支持以下子命令：

- **train**: 训练模型（可选配置文件）
- **enhance**: 增强单张图像或目录中的图像
- **create-data**: 创建示例数据（2D、3D或虚拟数据）
- **test**: 测试整个数据管道

通用选项：
- `--config`: 配置文件路径
- `--verbose` / `-v`: 启用详细输出
- `--log-dir`: 日志文件目录（默认：`./logs`）
- `--dry-run`: 执行空运行（不实际执行）

### train_advanced.py

高级训练脚本，支持广泛的超参数控制：

**训练参数：**
- `--epochs`: 训练轮数
- `--batch-size`: 批量大小
- `--learning-rate`: 学习率
- `--optimizer`: 优化器（Adam、AdamW、SGD）
- `--loss`: 损失函数（L1Loss、MSELoss、SSIMLoss、MixedLoss、MultiScaleLoss）
- `--scheduler`: 学习率调度器

**高级选项：**
- `--weight-decay`: 权重衰减（正则化）
- `--gradient-clip`: 梯度裁剪值
- `--warmup-epochs`: 学习率热身轮数
- `--patience`: 早停耐心值
- `--loss-weights`: 混合损失权重（逗号分隔）
- `--multi-scale`: 启用多尺度损失
- `--multi-scale-weights`: 多尺度损失权重

**模型参数：**
- `--model`: 模型架构（UNet2D、WaveletDomainCNN、FBPConvNet、MultiScaleModel）
- `--features`: 特征通道数（逗号分隔）

**数据参数：**
- `--data-dir`: 数据目录
- `--image-size`: 图像尺寸（逗号分隔）

**实验管理：**
- `--experiment`: 实验名称（用于组织检查点和日志）
- `--resume`: 从检查点恢复训练

### create_sample_data.py

创建用于测试和开发的合成CT数据：

```bash
# 创建2D CT样本
python scripts/data/create_sample_data.py --type 2d --num 20

# 创建3D CT体积
python scripts/data/create_sample_data.py --type 3d --num 5

# 创建虚拟数据
python scripts/data/create_sample_data.py --type dummy --num 50

# 测试数据管道
python scripts/data/create_sample_data.py --type test
```

## 工具模块

### logging.py

提供日志配置：

```python
from scripts.utils.logging import get_script_logger

logger = get_script_logger("my_script")
logger.info("信息")
logger.error("错误信息")
```

### config_loader.py

配置管理工具：

```python
from scripts.utils.config_loader import load_yaml_config, save_yaml_config

config = load_yaml_config("config.yaml")
save_yaml_config(config, "config_backup.yaml")
```

### error_handler.py

错误处理工具：

```python
from scripts.utils.error_handler import handle_exceptions, ScriptError

@handle_exceptions(log_error=True, exit_on_error=True)
def my_function():
    # 可能引发异常的代码
    pass
```

### arg_parser.py

命令行参数解析：

```python
from scripts.utils.arg_parser import create_base_parser, add_training_args

parser = create_base_parser("我的脚本")
add_training_args(parser)
args = parser.parse_args()
```

## 日志记录

所有脚本在 `./logs` 目录（可通过 `--log-dir` 配置）中生成日志文件。日志文件以脚本名称命名（例如 `main.log`、`train_advanced.log`）。

日志格式：`YYYY-MM-DD HH:MM:SS - logger_name - LEVEL - message`

## 错误处理

脚本使用结构化错误处理，包含以下异常类型：
- `ScriptError`: 脚本错误基类
- `ConfigError`: 配置相关错误
- `DataError`: 数据相关错误
- `ModelError`: 模型相关错误

使用 `@handle_exceptions` 装饰的函数将记录错误并优雅退出。

## 配置文件

配置文件采用YAML格式。示例配置文件位于 `configs/advanced_training_config.yaml`。

脚本按以下顺序查找配置文件：
1. `--config` 参数指定的路径
2. `./configs/config.yaml`
3. `./configs/advanced_training_config.yaml`
4. `./config.yaml`

## 最佳实践

1. **常用任务使用main.py**：对于大多数操作，使用 `main.py` 及其子命令。
2. **实验使用train_advanced.py**：调整超参数时，使用 `train_advanced.py`。
3. **检查日志**：始终检查日志文件以获取详细执行信息。
4. **使用空运行**：在实际执行前使用 `--dry-run` 测试命令。
5. **组织实验**：使用 `--experiment` 标志组织检查点和日志。

## 故障排除

### 常见问题

1. **模块导入错误**：确保从项目根目录运行脚本。
2. **CUDA内存不足**：使用 `--batch-size` 减小批量大小。
3. **文件未找到**：检查路径并确保数据目录存在。
4. **配置错误**：验证配置文件中的YAML语法。

### 获取帮助

- 检查 `./logs` 目录中的日志文件
- 使用 `--verbose` 标志获取详细输出
- 使用 `--dry-run` 测试命令而不实际执行

## 未来改进

计划中的增强功能：
- [ ] 在 `inference/` 目录中添加推理专用脚本
- [ ] 添加数据增强脚本
- [ ] 添加模型评估脚本
- [ ] 添加超参数优化脚本
- [ ] 添加结果可视化脚本
