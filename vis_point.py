import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import open3d as o3d
# import torch.multiprocessing as mp
from utils.o3d_utils import set_view_params
import threading
from utils.render_utils import get_state_at_time

view_params = {	
    "front" : [ 0.13840387030698442, 0.37568463563818738, -0.91635442009598544 ],
    "lookat" : [ 0.80928201521528009, 2.2729272410174697, 2.1735352590240136 ],
    "up" : [ 0.077264919625831902, -0.92653510294458685, -0.36818858646986569 ],
    "zoom" : 0.11999999999999962
}

view_params = {	
    "front" : [ -0.74970178541036581, 0.25255019652837529, -0.61169079704209128 ],
    "lookat" : [ -1.3969642231011352, 1.5126668451990903, 4.5636690302264631 ],
    "up" : [ 0.2597874175170658, -0.7378048856167535, -0.62302042379031453 ],
    "zoom" : 0.11999999999999962
}

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    return gaussians, scene

def save_point_cloud(points, model_path, timestamp, format="ply"):
    output_path = os.path.join(model_path,"point_pertimestamp")
    if not os.path.exists(output_path):
        os.makedirs(output_path,exist_ok=True)

    points = points.detach().cpu().numpy()

    if format == "ply":
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        ply_path = os.path.join(output_path,f"points_{timestamp:05d}.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
    elif format == "npy":
        np.save(os.path.join(output_path,f"points_{timestamp:05d}.npy"), points)

parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
hyperparam = ModelHiddenParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--skip_video", action="store_true")
parser.add_argument("--configs", type=str)
args = get_combined_args(parser)
print("Rendering " , args.model_path)
if args.configs:
    import mmcv
    from utils.params_utils import merge_hparams
    config = mmcv.Config.fromfile(args.configs)
    args = merge_hparams(args, config)

# Initialize system state (RNG)
safe_state(args.quiet)

vis = o3d.visualization.Visualizer()
vis.create_window()

bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-6, -6, -6), max_bound=(6, 3, 6))
vis.add_geometry(bbox)

coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
vis.add_geometry(coord_frame)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
vis.add_geometry(pcd)

set_view_params(vis, view_params)

gaussians, scene = render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), 
                               args.skip_train, args.skip_test, args.skip_video)
for index, viewpoint in enumerate(scene.getVideoCameras()):
    points, scales_final, rotations_final, opacity_final, shs_final = get_state_at_time(gaussians, viewpoint)

    pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())

    vis.update_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(os.path.join(args.model_path, "points_deform_view2", 
                                          f"points_{index:05d}.png"))

    # save_point_cloud(points, args.model_path, index, format="npy")
