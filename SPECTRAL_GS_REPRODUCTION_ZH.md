# Spectral-GS 论文复现说明

本分支在原始 **3D Gaussian Splatting** 代码库基础上，实现 **Spectral-GS: Taming 3D Gaussian Splatting with Spectral Entropy** 的核心方法。

参考资料：
- 本地论文，如果存在：`paper.pdf`
- 项目主页：https://letianhuang.github.io/spectralgs/
- arXiv：https://arxiv.org/abs/2409.12771

## 原代码库离论文还差什么

之前的项目代码是在 3DGS 上加入了一个受 Spectral-GS 启发的图像频域方法：

- 对渲染图和真实图计算 DCT 图像频谱熵 loss。
- 用局部图像频谱熵差异指导 densification。
- 对投影到高频误差区域的 Gaussian 做 gradient boost。

这些改动不等于 faithful reproduction。Spectral-GS 论文的核心是分析每个 **3D Gaussian covariance matrix** 的 spectral entropy，而不是分析渲染图像的 DCT 频谱。因此原代码还缺：

- 每个 Gaussian 的 covariance spectral entropy、condition number、spectral radius。
- 对低 entropy、needle-like Gaussian 的 gradient-independent splitting。
- split 之后的 anisotropic scale reduction。
- zoom-in 渲染时的 2D view-consistent filtering。
- zoom-factor evaluation 和 Gaussian shape statistics。

## 本分支补齐了什么

本分支补齐了论文的两个核心组件：

- **3D covariance spectral metrics**：`utils/spectral_utils.py` 现在可以从 activated Gaussian scales 计算 spectral entropy、condition number 和 spectral radius。3DGS 中 covariance eigenvalues 等于 `scale ** 2`。
- **3D shape-aware splitting**：`scene/gaussian_model.py` 支持低 entropy Gaussian 不依赖 view-space gradient 直接 split。Spectral split 会对 dominant scale axis 做 anisotropic shrinkage。
- **2D view-consistent filtering**：CUDA rasterizer 可以把固定 EWA filter variance `0.3` 替换为 Spectral-GS 的 scalar approximation：`s0 * focal_length^2 / depth^2`。
- **Zoom evaluation**：`render.py` 支持 `--zoom_factor`，并对 ground truth 做中心裁剪和 resize，使其匹配 zoomed field of view。
- **Spectral statistics**：训练时记录 Gaussian entropy/condition statistics，`metrics.py` 会写入 `results_spectral.json`。

旧的 DCT 图像频域功能仍然保留，可以作为 optional ablation。报告中应把它标注为 inspired baseline，而不是 Spectral-GS 论文的 faithful reproduction。

## 训练环境要求

推荐硬件：

- NVIDIA GPU，建议 RTX 4090 24GB。
- CUDA 11.6 或 11.8 兼容的 driver/toolchain。
- 使用 `environment.yml` 创建 Conda 环境。

安装环境和 CUDA extensions：

```sh
conda env create --file environment.yml --name GS3d
conda activate GS3d
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/fused-ssim
```

建议 final experiments 至少使用：

- Blender synthetic scenes：`lego`、`chair`
- Tanks and Temples real scene：`truck`

数据需要符合 3DGS `train.py` 期望的目录结构。Real scenes 需要 COLMAP 结果，synthetic scenes 使用 Blender transforms。

## 训练命令

原始 3DGS baseline：

```sh
python train.py -s data/lego -m output/lego_3dgs --eval
```

完整 Spectral-GS 复现版本：

```sh
python train.py -s data/lego -m output/lego_spectral \
  --eval \
  --use_spectral_shape_split \
  --use_view_consistent_filter \
  --spectral_entropy_threshold 0.5 \
  --spectral_delta 0.6 \
  --spectral_k0 1.0 \
  --spectral_split_N 2 \
  --spectral_filter_s0 0.1
```

可选的旧 DCT-inspired baseline：

```sh
python train.py -s data/lego -m output/lego_dct_inspired \
  --eval \
  --use_spectral_loss \
  --use_spectral_densification
```

渲染标准视角和 zoomed test views：

```sh
python render.py -m output/lego_spectral --eval --skip_train --zoom_factor 1
python render.py -m output/lego_spectral --eval --skip_train --zoom_factor 2
python render.py -m output/lego_spectral --eval --skip_train --zoom_factor 4
```

计算指标：

```sh
python metrics.py -m output/lego_3dgs output/lego_spectral
```

`metrics.py` 会写出：

- `results.json`
- `results_spectral.json`
- `per_view.json`

## RTX 4090 训练时间预估

30k iterations 的大致时间：

- Blender scene：每个 method 每个 scene 约 20-40 分钟。
- Real scene，例如 Truck：每个 method 每个 scene 约 45-90 分钟。
- 3 个 scenes x 2 个 methods 的主实验：约 3-6 GPU 小时。
- 在一个 scene 上做 ablation：额外约 1-2 GPU 小时。

实际时间会受 image resolution、input views 数量、CUDA build、是否使用 `--resolution` 降采样、是否有 accelerated rasterizer/optimizer 影响。

快速调试命令：

```sh
python train.py -s data/lego -m output/debug_spectral \
  --eval \
  --iterations 1000 \
  --resolution 8 \
  --use_spectral_shape_split \
  --use_view_consistent_filter
```

## 实验设计

主对比实验：

- Scenes：`lego`、`chair`、`truck`。
- Methods：原始 3DGS baseline、可选 DCT-inspired baseline、完整 Spectral-GS。
- Metrics：PSNR、SSIM、LPIPS、mean spectral entropy、mean/max condition number、Gaussian count。
- Zoom factors：`1x`、`2x`、`4x`；如果时间允许，再做 `8x`。

一个 scene 上的 ablation，推荐 `lego` 或 `chair`：

- Baseline 3DGS。
- Shape-aware split only：只开 `--use_spectral_shape_split`。
- View-consistent filter only：只开 `--use_view_consistent_filter`。
- Full method：两个 flag 都开。
- Optional DCT-inspired version：`--use_spectral_loss --use_spectral_densification`。

报告中的 qualitative outputs：

- 高频区域的 side-by-side crops。
- Zoom-in comparison，用来展示 needle-like artifacts。
- 表格记录 runtime、final Gaussian count、entropy、condition number、PSNR、SSIM、LPIPS。
- 解释 higher entropy 和 lower condition number 表示 needle-like Gaussians 更少。

## 验证清单

正式实验前先跑：

```sh
python -m py_compile train.py render.py metrics.py arguments/__init__.py scene/gaussian_model.py utils/spectral_utils.py
pip install -e submodules/diff-gaussian-rasterization
python -c "from diff_gaussian_rasterization import GaussianRasterizer; print('rasterizer import ok')"
```

然后跑 smoke test：

```sh
python train.py -s data/lego -m output/smoke_3dgs --eval --iterations 1000 --resolution 8
python train.py -s data/lego -m output/smoke_spectral --eval --iterations 1000 --resolution 8 --use_spectral_shape_split --use_view_consistent_filter
python render.py -m output/smoke_spectral --eval --skip_train --zoom_factor 2
python metrics.py -m output/smoke_3dgs output/smoke_spectral
```

预期结果：

- 新 flag 全部关闭时，baseline 仍然能正常训练。
- Spectral mode 能训练和渲染。
- `results_spectral.json` 包含图像指标和 Gaussian spectral statistics。
- Zoom render 结果会出现在 `test/ours_<iteration>_zoom2/`。
