# 3D 高斯泼溅 + Spectral-GS 改进版

> COMP5405 课程项目 | 基于 [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) 原始代码，融入 Spectral-GS 频谱熵引导优化方法。

## 项目简介

本项目在标准 3D 高斯泼溅（3DGS）框架基础上，参考论文  
**"Spectral-GS: Taming 3D Gaussian Splatting with Spectral Entropy"**  
引入频域感知机制，通过谱熵损失和频率引导致密化，改善渲染质量，减少针状伪影与模糊问题。

---

## 主要改动

| 文件 | 改动内容 |
|---|---|
| `utils/spectral_utils.py` | **新增**：2D DCT、谱熵计算、谱损失函数、局部谱熵图 |
| `arguments/__init__.py` | 新增 8 个 Spectral-GS 超参数 |
| `train.py` | 接入谱熵损失 + 谱引导致密化统计 |
| `scene/gaussian_model.py` | 新增谱累积器、谱引导致密化逻辑，兼容原有检查点格式 |

### 核心创新（来自 Spectral-GS 论文）

1. **谱熵损失**：对渲染图与真实图分别计算 2D DCT 功率谱熵，惩罚频率分布差异
2. **谱引导致密化**：在频域差异大的区域额外触发 Gaussian 分裂/克隆
3. **所有新功能默认关闭**，不加参数时行为与原版完全一致

---

## 环境配置

### 前置要求

- Windows 10/11，NVIDIA GPU（建议 8GB+ 显存）
- CUDA 11.x（与 PyTorch 1.12 对应）
- Visual Studio 2019/2022（含 C++ 桌面开发工作负载）
- Conda（Miniconda 或 Anaconda）
- COLMAP

### 创建环境

```bash
# 创建名为 GS3d 的 conda 环境
conda env create --file environment.yml --name GS3d
conda activate GS3d
```

---

## 数据准备

将视频帧放入 `data/input/`，然后运行 COLMAP 转换：

```bash
python convert.py -s data --colmap_executable "path/to/COLMAP.bat"
```

---

## 训练

### 标准 3DGS 训练（原版）

```bash
python train.py -s data -m data/output
```

### Spectral-GS 训练（改进版）

```bash
python train.py -s data -m data/output_spectral --use_spectral_loss --use_spectral_densification
```

### 常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--use_spectral_loss` | False | 开启谱熵损失 |
| `--use_spectral_densification` | False | 开启谱引导致密化 |
| `--lambda_spectral_entropy` | 0.01 | 谱熵差损失权重 |
| `--lambda_spectral_l1` | 0.01 | DCT 系数 L1 损失权重 |
| `--spectral_loss_from_iter` | 500 | 谱损失开始迭代数 |
| `--spectral_densify_threshold` | 0.5 | 谱引导致密化触发阈值 |
| `--resolution` | -1（原始） | 图片缩放比例（4 = 1/4 分辨率） |

---

## 渲染与评估

```bash
# 渲染图片
python render.py -m data/output --eval
python render.py -m data/output_spectral --eval

# 计算 PSNR / SSIM / LPIPS 指标
python metrics.py -m data/output
python metrics.py -m data/output_spectral
```

---

## 可视化

使用内置 SIBR Viewer：

```cmd
cd /d "viewers\bin"
SIBR_gaussianViewer_app.exe -m "path\to\output"
```

SIBR Viewer 基本操作：

| 操作 | 方式 |
|---|---|
| 旋转视角 | 鼠标左键拖动 |
| 平移 | 鼠标中键拖动 |
| 前进/后退 | W / S |
| 截图 | P |

---

## 项目结构

```
gaussian-splatting/
├── train.py                  # 训练主脚本（已修改）
├── render.py                 # 渲染脚本
├── metrics.py                # 指标评估脚本
├── convert.py                # COLMAP 数据转换脚本
├── arguments/__init__.py     # 参数定义（已修改，新增谱参数）
├── scene/
│   └── gaussian_model.py     # 高斯模型（已修改，加入谱致密化）
├── utils/
│   ├── spectral_utils.py     # 新增：谱工具函数
│   └── loss_utils.py         # 损失函数
├── gaussian_renderer/        # 渲染器
└── submodules/               # CUDA 扩展（diff-gaussian-rasterization 等）
```

---

## 参考文献

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)  
  Kerbl et al., SIGGRAPH 2023

- Spectral-GS: Taming 3D Gaussian Splatting with Spectral Entropy

---

## 开发环境

- Python 3.7 / PyTorch 1.12.1 / CUDA 11.6
- Windows 11 Pro
