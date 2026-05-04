# Spectral-GS Paper Reproduction

This branch implements the core components of **Spectral-GS: Taming 3D Gaussian Splatting with Spectral Entropy** on top of the original 3D Gaussian Splatting codebase.

Reference material:
- Local paper, if present: `paper.pdf`
- Project page: https://letianhuang.github.io/spectralgs/
- arXiv: https://arxiv.org/abs/2409.12771

## What Was Missing

The previous project code was based on 3DGS and added an image-frequency idea inspired by Spectral-GS:

- DCT-based image spectral entropy loss between rendered and ground-truth images.
- Local image spectral entropy maps for frequency-guided densification.
- Gradient boosting for Gaussians projected into high image-frequency-error regions.

That was not a faithful reproduction of the Spectral-GS paper. The paper's key idea is spectral analysis of each **3D Gaussian covariance matrix**, not DCT entropy of the rendered image. The previous code was missing:

- Per-Gaussian covariance spectral entropy, condition number, and spectral radius.
- Gradient-independent splitting for low-entropy, needle-like Gaussians.
- Anisotropic scale reduction after splitting.
- View-consistent 2D filtering for zoom-in rendering.
- Zoom-factor evaluation and Gaussian shape statistics.

## What This Branch Adds

This branch adds the two core paper components:

- **3D covariance spectral metrics**: `utils/spectral_utils.py` now computes spectral entropy, condition number, and spectral radius from activated Gaussian scales. In 3DGS, covariance eigenvalues are `scale ** 2`.
- **3D shape-aware splitting**: `scene/gaussian_model.py` now supports low-entropy Gaussian splitting independent of view-space gradient. Spectral splits use anisotropic shrinkage on the dominant scale axis.
- **2D view-consistent filtering**: the CUDA rasterizer can replace the fixed EWA filter variance `0.3` with the Spectral-GS scalar approximation `s0 * focal_length^2 / depth^2`.
- **Zoom evaluation**: `render.py` supports `--zoom_factor`, and generated ground truth is center-cropped and resized to match the zoomed field of view.
- **Spectral statistics**: training logs Gaussian entropy/condition statistics, and `metrics.py` writes them into `results_spectral.json`.

The old DCT image-frequency features are still present as optional ablations. They should be reported as an inspired baseline, not as the faithful Spectral-GS reproduction.

## Training Requirements

Recommended hardware:

- NVIDIA GPU, ideally RTX 4090 24GB.
- CUDA 11.6 or 11.8 compatible driver/toolchain.
- Conda environment from `environment.yml`.

Install environment and CUDA extensions:

```sh
conda env create --file environment.yml --name GS3d
conda activate GS3d
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/fused-ssim
```

Minimum datasets for final experiments:

- Blender synthetic scenes: `lego`, `chair`
- Tanks and Temples real scene: `truck`

Use the standard 3DGS dataset layout expected by `train.py`, with COLMAP data for real scenes and Blender transforms for synthetic scenes.

## Training Commands

Baseline 3DGS:

```sh
python train.py -s data/lego -m output/lego_3dgs --eval
```

Full Spectral-GS reproduction:

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

Optional old DCT-inspired baseline:

```sh
python train.py -s data/lego -m output/lego_dct_inspired \
  --eval \
  --use_spectral_loss \
  --use_spectral_densification
```

Render standard and zoomed test views:

```sh
python render.py -m output/lego_spectral --eval --skip_train --zoom_factor 1
python render.py -m output/lego_spectral --eval --skip_train --zoom_factor 2
python render.py -m output/lego_spectral --eval --skip_train --zoom_factor 4
```

Compute metrics:

```sh
python metrics.py -m output/lego_3dgs output/lego_spectral
```

`metrics.py` writes:

- `results.json`
- `results_spectral.json`
- `per_view.json`

## RTX 4090 Time Estimate

Approximate runtime for 30k iterations:

- Blender scene: 20-40 minutes per method per scene.
- Real scene such as Truck: 45-90 minutes per method per scene.
- Main experiment with 3 scenes x 2 methods: about 3-6 GPU hours.
- Ablations on one scene: about 1-2 extra GPU hours.

Runtime depends on image resolution, number of input views, CUDA build speed, use of `--resolution`, and whether the accelerated rasterizer/optimizer is available.

For quick debugging:

```sh
python train.py -s data/lego -m output/debug_spectral \
  --eval \
  --iterations 1000 \
  --resolution 8 \
  --use_spectral_shape_split \
  --use_view_consistent_filter
```

## Experiment Design

Main comparison:

- Scenes: `lego`, `chair`, `truck`.
- Methods: original 3DGS baseline, optional DCT-inspired baseline, full Spectral-GS.
- Metrics: PSNR, SSIM, LPIPS, mean spectral entropy, mean/max condition number, Gaussian count.
- Zoom factors: `1x`, `2x`, `4x`; use `8x` only if time allows.

Ablation on one scene, preferably `lego` or `chair`:

- Baseline 3DGS.
- Shape-aware split only: `--use_spectral_shape_split`.
- View-consistent filter only: `--use_view_consistent_filter`.
- Full method: both flags enabled.
- Optional DCT-inspired version: `--use_spectral_loss --use_spectral_densification`.

Qualitative outputs for the report:

- Side-by-side crops of high-frequency regions.
- Zoom-in comparisons showing needle-like artifacts.
- Table with runtime, final Gaussian count, entropy, condition number, PSNR, SSIM, LPIPS.
- A short explanation that higher entropy and lower condition number indicate fewer needle-like Gaussians.

## Verification Checklist

Before running final experiments:

```sh
python -m py_compile train.py render.py metrics.py arguments/__init__.py scene/gaussian_model.py utils/spectral_utils.py
pip install -e submodules/diff-gaussian-rasterization
python -c "from diff_gaussian_rasterization import GaussianRasterizer; print('rasterizer import ok')"
```

Then run:

```sh
python train.py -s data/lego -m output/smoke_3dgs --eval --iterations 1000 --resolution 8
python train.py -s data/lego -m output/smoke_spectral --eval --iterations 1000 --resolution 8 --use_spectral_shape_split --use_view_consistent_filter
python render.py -m output/smoke_spectral --eval --skip_train --zoom_factor 2
python metrics.py -m output/smoke_3dgs output/smoke_spectral
```

Expected outcomes:

- Baseline still trains with all new flags off.
- Spectral mode trains and renders.
- `results_spectral.json` contains image metrics and Gaussian spectral statistics.
- Zoom render folders appear under `test/ours_<iteration>_zoom2/`.
