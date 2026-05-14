# Spectral-GS Reproduction for Zoom-Robust 3D Gaussian Splatting

This repository is the COMP5405/4405 final project submission for reproducing the core ideas of **Spectral-GS: Taming 3D Gaussian Splatting with Spectral Entropy** on top of a 3D Gaussian Splatting codebase.

The project targets zoom-robust novel-view synthesis. Standard 3D Gaussian Splatting can produce elongated, needle-like Gaussian primitives that fit training views but become unstable when rendered under changed sampling rates such as zoom-in views. Spectral-GS addresses this by analyzing each Gaussian covariance spectrum and by adapting the 2D rasterization filter.

This implementation should be described as a faithful core reproduction, not a complete paper-level reproduction. It implements the main 3D shape-aware splitting mechanism and a practical scalar view-consistent filtering approximation, but it does not reproduce the full matrix filter or the full benchmark scale of the paper.

## Implemented Components

- **3D covariance spectral metrics**: spectral entropy, condition number, and spectral radius are computed from activated Gaussian scales. Since the 3DGS covariance is parameterized as `Sigma = R S S^T R^T`, the covariance eigenvalues are `scale ** 2`.
- **Low-entropy shape-aware splitting**: Gaussians with spectral entropy below the threshold can be split independently of the standard view-space gradient trigger.
- **Anisotropic dominant-axis shrink**: spectral-selected splits shrink the largest scale axis more strongly to reduce needle-like Gaussian shapes.
- **Scalar view-consistent 2D filtering**: the rasterizer can replace the fixed EWA variance with `s = s0 * f^2 / z^2`, where `f` is average focal length and `z` is camera-space depth.
- **Zoom-factor rendering**: `render.py` supports `--zoom_factor`; ground-truth images are center-cropped and resized to match the zoomed field of view.
- **Metric and shape reporting**: `metrics.py` reports PSNR, SSIM, LPIPS, and Gaussian spectral statistics when point-cloud checkpoints are available.

## Main Limitations

- The full matrix view-consistent filter from the paper is not implemented. The current implementation uses the scalar approximation `s = s0 * f^2 / z^2`.
- The final experiments use 15000 training iterations, not the full 30000-iteration paper setting.
- The evaluation covers four scenes, not the full 12-scene benchmark used by the paper.
- The final ablation includes split-only results, but does not include a separate filter-only ablation.
- The inherited repository still contains older DCT-inspired spectral loss and densification code. These options are kept as optional legacy ablations, but they are not the final Spectral-GS method reported in the project.

## Environment Requirements

Recommended hardware:

- NVIDIA GPU, ideally RTX 4090 24GB.
- CUDA 11.6 or 11.8 compatible driver and build toolchain.
- Conda environment from `environment.yml`.

Create the environment and install the CUDA extensions:

```sh
conda env create --file environment.yml --name GS3d
conda activate GS3d
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/fused-ssim
```

The exact CUDA/PyTorch setup depends on the local machine. If the rasterizer fails to import after installation, rebuild `submodules/diff-gaussian-rasterization` in the active environment.

## Dataset Layout

Datasets are not included in this repository or submission because of size. The final report uses four scenes:

- `truck`
- `train`
- `drjohnson`
- `playroom`

Expected layout:

```text
data/
  truck/
  train/
  drjohnson/
  playroom/
```

Real scenes should follow the standard 3DGS dataset format with COLMAP camera data. Synthetic scenes, if used for extra testing, should follow the Blender-style transform format expected by the original 3DGS loader.

## Training

All final reported experiments use image downsampling `-r 2` and 15000 iterations.

Baseline 3D-GS:

```sh
python train.py -s data/truck -m output/truck_baseline_r2 -r 2 --eval --iterations 15000
```

Split-only ablation:

```sh
python train.py -s data/truck -m output/truck_split_r2 -r 2 --eval --iterations 15000 --use_spectral_shape_split
```

Full Spectral-GS reproduction:

```sh
python train.py -s data/truck -m output/truck_full_r2 -r 2 --eval --iterations 15000 --use_spectral_shape_split --use_view_consistent_filter --spectral_filter_s0 0.0001
```

The default spectral split parameters are:

```text
spectral_entropy_threshold = 0.5
spectral_delta = 0.6
spectral_k0 = 1.0
spectral_split_N = 2
```

To run the same settings on another scene, replace `truck` with `train`, `drjohnson`, or `playroom` in both the source path and model path.

## Rendering and Metrics

Render standard and zoomed test views:

```sh
python render.py -m output/truck_full_r2 --eval --skip_train --zoom_factor 1
python render.py -m output/truck_full_r2 --eval --skip_train --zoom_factor 2
python render.py -m output/truck_full_r2 --eval --skip_train --zoom_factor 4
```

Compute metrics:

```sh
python metrics.py -m output/truck_baseline_r2 output/truck_split_r2 output/truck_full_r2
```

The metric script reports:

- PSNR: higher is better.
- SSIM: higher is better.
- LPIPS: lower is better.
- EntropyMean: mean 3D covariance spectral entropy of saved Gaussians. Higher values generally indicate less needle-like shape degeneration, but this is not an image-quality metric by itself.

## Final Results Summary

The final report compares the baseline 3D-GS model with the full Spectral-GS reproduction on `truck`, `train`, `drjohnson`, and `playroom`.

| Metric | Zoom | Baseline | Spectral-GS | Difference |
| --- | --- | ---: | ---: | ---: |
| PSNR up | 1x | **26.95** | 26.37 | -0.58 |
| PSNR up | 2x | 26.10 | **27.01** | +0.91 |
| PSNR up | 4x | 26.47 | **28.06** | +1.59 |
| SSIM up | 1x | **0.903** | 0.875 | -0.028 |
| SSIM up | 2x | 0.848 | **0.866** | +0.017 |
| SSIM up | 4x | 0.859 | **0.896** | +0.037 |
| LPIPS down | 1x | **0.134** | 0.174 | +0.040 |
| LPIPS down | 2x | 0.181 | **0.176** | -0.004 |
| LPIPS down | 4x | 0.214 | **0.187** | -0.027 |

At the original 1x scale, the baseline is slightly better on average. At 2x and 4x zoom, the reproduced Spectral-GS branch is stronger. At 4x zoom, it improves average PSNR by 1.59 dB, improves SSIM by 0.037, and reduces LPIPS by 0.027.

Detailed tables, qualitative figures, methodology explanation, limitations, and AI reflection are in:

- `report/Report/final_report.pdf`
- `report/Report/AI_USAGE_RECORD.pdf`

Detailed result data and qualitative crops are in:

- `result/results_comparison.csv`
- `result/playroom_zoom4/`
- `result/truck_zoom4/`

## Repository Structure

```text
train.py                                  Training entry point
render.py                                 Standard and zoom-factor rendering
metrics.py                                PSNR, SSIM, LPIPS, and spectral statistics
arguments/__init__.py                     New Spectral-GS command-line flags
scene/gaussian_model.py                   Shape-aware split and anisotropic shrink logic
utils/spectral_utils.py                   Covariance spectral metric utilities
gaussian_renderer/                        Python renderer interface
submodules/diff-gaussian-rasterization/   CUDA rasterizer with scalar view-consistent filter
report/Report/                            Final report and AI usage record
result/                                   Final result tables and qualitative crops
```

## Verification Checklist

Run a Python syntax check:

```sh
python -m py_compile train.py render.py metrics.py arguments/__init__.py scene/gaussian_model.py utils/spectral_utils.py
```

Check that the CUDA rasterizer imports:

```sh
python -c "from diff_gaussian_rasterization import GaussianRasterizer; print('rasterizer import ok')"
```

Optional smoke test with a small scene:

```sh
python train.py -s data/truck -m output/smoke_spectral -r 8 --eval --iterations 1000 --use_spectral_shape_split --use_view_consistent_filter --spectral_filter_s0 0.0001
python render.py -m output/smoke_spectral --eval --skip_train --zoom_factor 2
python metrics.py -m output/smoke_spectral
```

Expected outcomes:

- Baseline still trains when all Spectral-GS flags are disabled.
- Spectral mode trains and renders when the new flags are enabled.
- Zoom render folders are generated for requested zoom factors.
- Metric output includes image metrics and, when available, Gaussian spectral statistics.

## Submission Files

The main submission-facing files are:

- `README.md`
- `report/Report/final_report.pdf`
- `report/Report/AI_USAGE_RECORD.pdf`
- Source code in this repository
- Final qualitative and quantitative results under `result/`
