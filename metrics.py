#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import numpy as np
from plyfile import PlyData


def gaussian_spectral_stats(scene_dir):
    point_cloud_dir = Path(scene_dir) / "point_cloud"
    if not point_cloud_dir.exists():
        return {}
    iteration_dirs = []
    for path in point_cloud_dir.iterdir():
        if path.is_dir() and path.name.startswith("iteration_"):
            try:
                iteration_dirs.append((int(path.name.split("_")[-1]), path))
            except ValueError:
                pass
    if not iteration_dirs:
        return {}

    _, latest_dir = max(iteration_dirs, key=lambda item: item[0])
    ply_path = latest_dir / "point_cloud.ply"
    if not ply_path.exists():
        return {}

    plydata = PlyData.read(ply_path)
    scale_names = sorted(
        [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if len(scale_names) != 3:
        return {}

    log_scales = np.stack([np.asarray(plydata.elements[0][name]) for name in scale_names], axis=1)
    scales = np.exp(log_scales)
    eigvals = np.maximum(scales ** 2, 1e-12)
    probs = eigvals / np.maximum(eigvals.sum(axis=1, keepdims=True), 1e-12)
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    condition = eigvals.max(axis=1) / np.maximum(eigvals.min(axis=1), 1e-12)
    return {
        "EntropyMean": float(entropy.mean()),
        "EntropyMedian": float(np.median(entropy)),
        "EntropyMin": float(entropy.min()),
        "ConditionMean": float(condition.mean()),
        "ConditionMax": float(condition.max()),
        "GaussianCount": int(scales.shape[0]),
    }


def zoom_factor_from_method(method):
    if "_zoom" not in method:
        return 1.0
    try:
        return float(method.rsplit("_zoom", 1)[-1])
    except ValueError:
        return 1.0

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            spectral_stats = gaussian_spectral_stats(scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in sorted(os.listdir(test_dir)):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                method_metrics = {"SSIM": torch.tensor(ssims).mean().item(),
                                  "PSNR": torch.tensor(psnrs).mean().item(),
                                  "LPIPS": torch.tensor(lpipss).mean().item(),
                                  "ZoomFactor": zoom_factor_from_method(method)}
                method_metrics.update(spectral_stats)
                full_dict[scene_dir][method].update(method_metrics)
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/results_spectral.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as exc:
            print("Unable to compute metrics for model", scene_dir, "because", repr(exc))

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
