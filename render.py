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

import torch
from scene import Scene
import os
import math
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import torch.nn.functional as F
from utils.general_utils import safe_state
from utils.graphics_utils import getProjectionMatrix
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def _zoom_fov(fov, zoom_factor):
    return 2.0 * math.atan(math.tan(fov * 0.5) / zoom_factor)


def _zoom_ground_truth(gt, zoom_factor):
    if zoom_factor == 1.0:
        return gt
    _, height, width = gt.shape
    crop_h = max(1, int(round(height / zoom_factor)))
    crop_w = max(1, int(round(width / zoom_factor)))
    top = max(0, (height - crop_h) // 2)
    left = max(0, (width - crop_w) // 2)
    cropped = gt[:, top:top + crop_h, left:left + crop_w]
    return F.interpolate(cropped.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0)


def _render_zoomed(view, gaussians, pipeline, background, train_test_exp, separate_sh, zoom_factor):
    if zoom_factor == 1.0:
        return render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]

    old_fovx, old_fovy = view.FoVx, view.FoVy
    old_projection_matrix = view.projection_matrix
    old_full_proj_transform = view.full_proj_transform
    try:
        view.FoVx = _zoom_fov(old_fovx, zoom_factor)
        view.FoVy = _zoom_fov(old_fovy, zoom_factor)
        view.projection_matrix = getProjectionMatrix(
            znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy
        ).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        return render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
    finally:
        view.FoVx, view.FoVy = old_fovx, old_fovy
        view.projection_matrix = old_projection_matrix
        view.full_proj_transform = old_full_proj_transform


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, zoom_factor):
    method_name = "ours_{}".format(iteration) if zoom_factor == 1.0 else "ours_{}_zoom{:g}".format(iteration, zoom_factor)
    render_path = os.path.join(model_path, name, method_name, "renders")
    gts_path = os.path.join(model_path, name, method_name, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = _render_zoomed(view, gaussians, pipeline, background, train_test_exp, separate_sh, zoom_factor)
        gt = _zoom_ground_truth(view.original_image[0:3, :, :], zoom_factor)

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, zoom_factor: float):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, zoom_factor)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, zoom_factor)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--zoom_factor", default=1.0, type=float)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args.zoom_factor)
