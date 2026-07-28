# HUST-SOFT 触觉杨氏模量估计代码

[English README](README.md)

本仓库提供 HUST-SOFT 数据集上的触觉杨氏模量估计模型训练与评估代码。模型以
三帧时序触觉图像和对应的接触力为输入，主要包含 ResNet-18 图像编码器、物理信息
引导的交叉注意力、空间可变形卷积、时间卷积与动态门控模块，以及杨氏模量回归头。

## 1. 使用流程

所有命令都应在仓库根目录执行。

```bash
# 创建环境
conda create -n hust-soft python=3.10 -y
conda activate hust-soft

# 安装依赖
pip install -r requirements.txt

# 生成演示数据并检查程序
python scripts/create_demo_dataset.py
python train.py --config configs/demo.yaml --dry-run

# 完整运行 1 个 epoch 的演示实验
python train.py --config configs/demo.yaml
```

演示实验运行结束后，`outputs/demo/` 中应当出现模型权重、训练指标、回归结果图和
混淆矩阵。这组演示数据仅用于检查代码能否正常执行，不能用于论文实验或性能比较。

## 2. 仓库结构

```text
.
├── configs/
│   ├── default.yaml                 # HUST-SOFT 正式训练配置
│   └── demo.yaml                    # 演示数据的一轮训练配置
├── data/
│   └── README.md                    # 数据放置说明
├── scripts/
│   └── create_demo_dataset.py       # 演示数据生成脚本
├── tests/
│   ├── test_data_pipeline.py        # 数据接口测试
│   └── test_model.py                # 模型前向测试
├── BioSyS.py # 模型结构
├── train.py                         # 推荐使用的训练入口
├── train_BioSyS.py                  # 原文件名兼容入口
├── requirements.txt                 # 运行依赖
└── requirements-dev.txt             # 测试与代码检查依赖
```

`train.py` 是整理后的正式入口。保留
`train_BioSyS.py` 是为了兼容原有运行命令，两者最终调用的是同一套
训练流程。

## 3. 环境配置

推荐使用 Python 3.10 或更高版本，并在独立 Conda 环境中运行：

```bash
conda create -n hust-soft python=3.10 -y
conda activate hust-soft
pip install -r requirements.txt
```

模型使用了 `torchvision.ops.DeformConv2d`，因此 `torch` 与 `torchvision`
必须相互兼容。如果需要指定 CUDA 版本，建议先通过
[PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)
生成对应的安装命令，再执行：

```bash
pip install -r requirements.txt
```

Weights & Biases 不是必要依赖。只有在需要在线记录实验时才安装：

```bash
pip install wandb
```

## 4. 准备 HUST-SOFT 数据集

数据集地址：
[HUST-SOFT on Hugging Face](https://huggingface.co/datasets/HUST-XZR/HUST-SOFT)

下载后，默认应将 CSV 文件和图像文件夹放置在 `data/` 目录下：

```text
data/
├── tactile_dataset.csv
└── 图像文件夹/
    ├── image_0001.png
    ├── image_0002.png
    └── ...
```

实际图像子目录可以不同，但 CSV 中的 `image_path` 必须能够相对于 `data/`
定位到对应图像。例如：

```csv
group_id,image_path,force_n,youngs_modulus_mpa,shore_a
sample_0001,HUST_SOFT1/angle/image_0001.png,10.0,1.19,31
sample_0001,HUST_SOFT1/angle/image_0002.png,20.0,1.19,31
sample_0001,HUST_SOFT1/angle/image_0003.png,30.0,1.19,31
```

各字段含义如下：

| 字段 | 含义 |
| --- | --- |
| `group_id` | 同一次时序按压过程的样本编号 |
| `image_path` | 相对于 `data.root` 的图像路径 |
| `force_n` | 接触力，单位为 N |
| `youngs_modulus_mpa` | 杨氏模量标签，单位为 MPa |
| `shore_a` | Shore A 硬度，用于分组绘图 |

程序会按照 `force_n` 对同一个 `group_id` 内的图像排序，并默认选取力最大的三帧作为
一个时序样本。训练集与验证集按照 `group_id` 进行划分，因此同一时序样本的不同帧
不会同时进入训练集和验证集。

Windows 路径分隔符 `\` 和 Linux 路径分隔符 `/` 均可识别。

## 5. 修改正式训练配置

正式训练前打开 `configs/default.yaml`。通常需要检查以下参数：

```yaml
data:
  root: ./data
  csv_filename: tactile_dataset.csv
  n_frames: 3
  image_height: 250
  image_width: 350
  validation_ratio: 0.20
  max_force: 45.0
  min_modulus: 0.0
  max_modulus: 18.0

training:
  seed: 27
  device: auto
  epochs: 80
  batch_size: 64
  learning_rate: 0.0001

output:
  root: ./outputs
  run_name: hust_soft_default
```

主要参数说明：

| 参数 | 说明 |
| --- | --- |
| `data.root` | CSV 与图像的根目录 |
| `data.csv_filename` | 数据标注 CSV 文件名 |
| `data.n_frames` | 每个样本使用的时序帧数 |
| `data.validation_ratio` | 验证集比例 |
| `data.max_force` | 接触力归一化上限 |
| `training.device` | `auto`、`cpu`、`cuda` 或 `cuda:0` |
| `training.batch_size` | 批大小，显存不足时应减小 |
| `output.run_name` | 本次实验的输出文件夹名称 |

如果 CSV 字段名与默认配置不同，可以修改 `data.columns`，不需要修改 Python 代码。

## 6. 正式训练前的数据自检

在开始长时间训练前，建议先运行：

```bash
python train.py --config configs/default.yaml --dry-run
```

该命令不会更新模型参数，而是依次检查：

1. CSV 文件是否存在以及必需字段是否完整；
2. CSV 中引用的所有图像是否能够找到；
3. 每个 `group_id` 是否能够组成三帧时序样本；
4. 训练集和验证集是否能够正常划分；
5. DataLoader 输出张量形状是否正确；
6. 模型能否完成一次前向计算。

只有出现 `Dry run passed.` 后，才建议启动正式训练。

## 7. 启动正式训练

使用默认配置：

```bash
python train.py --config configs/default.yaml
```

也可以直接通过命令行覆盖常用参数：

```bash
# 指定 GPU 并减小批大小
python train.py --device cuda:0 --batch-size 16

# 修改训练轮数和实验名称
python train.py --epochs 100 --run-name experiment_01

# 使用其他数据目录
python train.py --data-dir /path/to/HUST-SOFT

# 离线运行，不下载 ImageNet 预训练权重
python train.py --no-pretrained
```

默认情况下，ResNet-18 会使用 ImageNet 预训练权重。首次运行时可能需要联网下载。
如果实验必须使用预训练权重，不建议在下载失败时静默切换为随机初始化；如果仅用于
检查程序，可以使用 `--no-pretrained`。

## 8. 断点续训与 W&B

从最近一次检查点继续训练：

```bash
python train.py \
  --config configs/default.yaml \
  --resume outputs/hust_soft_default/last_checkpoint.pt
```

启用 W&B：

```bash
wandb login
python train.py --config configs/default.yaml --wandb
```

默认配置不会连接 W&B，因此没有账号或网络时也能正常训练。

## 9. 输出结果

每次实验结果保存在：

```text
outputs/<run_name>/
```

主要文件如下：

| 文件 | 内容 |
| --- | --- |
| `resolved_config.yaml` | 本次运行实际使用的完整配置 |
| `metrics.csv` | 每个 epoch 的 Loss、MAE、RMSE 和 MAPE |
| `last_checkpoint.pt` | 最近一个 epoch 的模型和优化器状态 |
| `best_checkpoint.pt` | 验证集 MAPE 最低时的完整检查点 |
| `best_model_state_dict.pt` | 仅包含最佳模型参数 |
| `regression_plot.png` | 真实杨氏模量与预测值的回归散点图 |
| `confusion_matrix.png` | 验证集杨氏模量类别混淆矩阵 |
| `classification_metrics.json` | 分类准确率及混淆矩阵数值 |

训练结束后，程序会自动载入验证集 MAPE 最低的模型，并生成最终回归图和混淆矩阵。
混淆矩阵中的类别由预测值与预设 MPa 类别之间的最近距离确定，类别列表位于
`configs/default.yaml` 的 `evaluation.class_values_mpa`。

## 10. 运行测试

安装开发依赖：

```bash
pip install -r requirements-dev.txt
pytest -q
```

正常情况下应显示：

```text
3 passed
```

## 11. 常见问题

### 找不到 CSV

检查 `configs/default.yaml` 中的 `data.root` 和 `data.csv_filename`。

### 找不到图像

检查 CSV 中的 `image_path` 是否相对于 `data.root`，并先执行 `--dry-run` 查看首批
无法解析的路径。

### 预训练权重下载失败

保证网络能够访问 PyTorch 权重服务器，或在仅检查程序时添加：

```bash
python train.py --no-pretrained
```

### `torchvision::deform_conv2d` 或自定义算子报错

通常是 `torch` 与 `torchvision` 版本或 CUDA 构建不匹配。重新安装官方推荐的配套
版本，不要单独更新其中一个包。

### CUDA 显存不足

减小批大小，例如：

```bash
python train.py --batch-size 8
```

也可以先使用 CPU 检查数据和模型接口：

```bash
python train.py --device cpu --dry-run
```

## 12. 引用

论文正式发表后更新引用信息。
