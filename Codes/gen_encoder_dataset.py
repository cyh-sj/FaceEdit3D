# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

from ast import arg
import os
import re
import math
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm
import mrcfile
import copy
import PIL.Image
import cv2
import copy
import time
import json
import random

import legacy

from camera_utils import LookAtPoseSampler
from torch_utils import misc
from training.triplane import TriPlaneGenerator
import matplotlib.pyplot as plt

#from torchvision import utils
#----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

def gen_interp_video(G, outdir: str, seeds, shuffle_seed=None, w_frames=60*4, kind='cubic', grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, truncation_cutoff=14, cfg='FFHQ', image_mode='image', gen_shapes=False, device=torch.device('cuda'), ifstyle=False, **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    G.rendering_kwargs['ray_start']=2.75
    G.rendering_kwargs['ray_end']=3.4

    if num_keyframes is None:
        if len(seeds) % (grid_w*grid_h) != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = len(seeds) // (grid_w*grid_h)

    all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
    for idx in range(num_keyframes*grid_h*grid_w):
        all_seeds[idx] = seeds[idx % len(seeds)]

    if shuffle_seed is not None:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(all_seeds)

    camera_lookat_point = torch.tensor([0, 0., 0.2], device=device) if cfg == 'FFHQ' else torch.tensor([0, 0, 0], device=device)
    # camera_lookat_point = torch.tensor([0, 0.05, 0.05], device=device) if cfg == 'FFHQ' else torch.tensor([0, 0, 0], device=device)

    zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
    #zs = torch.from_numpy(np.stack([np.random.randn(G.z_dim) for seed in all_seeds])).to(device)

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.0], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(len(zs), 1)
    ws = G.mapping(z=zs, c=c, truncation_psi=psi, truncation_cutoff=truncation_cutoff)
    #_ = G.synthesis(ws[:1], c[:1]) # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

    # Render video.
    voxel_resolution = 512

    radius = 2.7
    angles = [0.0]#, 3.14/6, 3.14/4]

    all_w = []
    all_label = []


    from wrap_2D.controller_warp_2D import WarpController
    initail_ldks_path = "wrap/3D_ldks.txt"
    target_ldks_path = "wrap/3D_deformed_ldks.txt"
    ldks_src_3d = torch.from_numpy(np.loadtxt(initail_ldks_path)).unsqueeze(0).float()
    # ldks_tgt_3d = torch.from_numpy(np.loadtxt(target_ldks_path)).unsqueeze(0).float()

    for seed_idx, seed in enumerate(seeds):
        angle = random.uniform(-3.14/4, 3.14/4)
        pitch_range = random.uniform(-3.14/8, 3.14/8)
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        cam2world_pose = LookAtPoseSampler.sample(3.14/2 + angle, 3.14/2 + pitch_range ,camera_lookat_point, radius=radius, device=device)
        intrinsics = torch.tensor([[4.4652 , 0, 0.5], [0, 4.4652 , 0.5], [0, 0, 1]], device=device)
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        c = c[0:1]
        w = G.mapping(z, c, truncation_psi=psi, truncation_cutoff=truncation_cutoff)

        triplanes_1 = G.backbone.synthesis(ws=w)

        # print(triplanes_1.size())
        triplanes_1 = triplanes_1.view(len(triplanes_1), 3, 32, triplanes_1.shape[-2], triplanes_1.shape[-1])

        planes = copy.deepcopy(triplanes_1)

        ldks_tgt_3d = copy.deepcopy(ldks_src_3d)
        if (angle >= 3.14/ 5) or (angle <= -3.14/5):
            temp = random.uniform(-0.03, 0.05)
            ldks_tgt_3d[:, 3, 2] += temp
            ldks_tgt_3d[:, 3, 2] += temp * 0.4
        else:
            for i in range(2):
                temp = random.randint(0,27)
                if temp in [11,12,13,14,15,16,17,19,20,21,22,23,24,25,26]:
                    range_xy_max = 0.02
                elif temp in [2,3,4]:
                    range_xy_max = 0.0
                else:
                    range_xy_max = 0.05
                
                ldks_tgt_3d[:, temp, 0] += random.uniform(0-range_xy_max, range_xy_max)
                if ldks_tgt_3d[:, temp, 1] <= 0.5:
                    ldks_tgt_3d[:, temp, 1] += random.uniform(0-range_xy_max, 0)
                else:
                    ldks_tgt_3d[:, temp, 1] -= random.uniform(0-range_xy_max, 0)

        planes = planes.permute(0, 1, 3, 4, 2)

        warper = WarpController(len(ldks_tgt_3d[0]), 'cuda')
        planes[:,0,:,:,:] = warper(planes[:,0,:,:,:], ldks_src_3d[:,:,[0,1]], ldks_tgt_3d[:,:,[0,1]])
        # planes[:,2,:,:,:] = warper(planes[:,2,:,:,:], ldks_src_3d[:,:,[0,2]], ldks_tgt_3d[:,:,[0,2]])
        planes[:,1,:,:,:] = warper(planes[:,1,:,:,:], ldks_src_3d[:,:,[2,1]], ldks_tgt_3d[:,:,[2,1]])

        # planes[:,2,:,:,:] = planes[:,0,:,:,:]
        # planes[:,1,:,:,:] = planes[:,0,:,:,:]
        end_time = time.time()
        planes = planes.permute(0, 1, 4, 2, 3)

        cam2world_pose = LookAtPoseSampler.sample(3.14/2 + angle, 3.14/2 + pitch_range ,camera_lookat_point, radius=radius, device=device)
        #all_poses.append(cam2world_pose.squeeze().cpu().numpy())
        intrinsics = torch.tensor([[4.4652 , 0, 0.5], [0, 4.4652 , 0.5], [0, 0, 1]], device=device)
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        c = c[0:1]
        # print(c)

        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        G.rendering_kwargs['ray_start']=2.85 + radius - 3.3
        G.rendering_kwargs['ray_end']=3.9 + radius - 3.3

        ray_origins, ray_directions = G.ray_sampler(cam2world_matrix, intrinsics, G.neural_rendering_resolution)
        N, M, _ = ray_origins.shape

        feature_samples, depth_samples, _ = G.renderer(planes, G.decoder, ray_origins, ray_directions, G.rendering_kwargs)
        
        origin_feature, _, _ = G.renderer(triplanes_1, G.decoder, ray_origins, ray_directions, G.rendering_kwargs)
        H = W = G.neural_rendering_resolution

        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        origin_image = origin_feature.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()

        rgb_image = feature_image[:, :3]

        sr_image = G.superresolution(rgb_image, feature_image, w)
        sr_origin_image = G.superresolution(origin_image[:, :3], origin_image, w)



        img = (sr_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].cpu().numpy()

        ori_img = (sr_origin_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        ori_img = ori_img[0].cpu().numpy()


        PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/warp/seed{seed:04d}.png')
        PIL.Image.fromarray(ori_img, 'RGB').save(f'{outdir}/origin/seed{seed:04d}.png')

        label = [f'{outdir}/origin/seed{seed:04d}.png', f'{outdir}/warp/seed{seed:04d}.png', c.cpu().numpy().tolist()]
        all_label.append(label)
        all_w.append(w.cpu().numpy())
    
    all_w = np.array(all_w)
    np.save('Dataset_encoder/w_codes.npy', all_w)
    with open('Dataset_encoder/labels.json', 'w') as f:
        json.dump(all_label, f, indent=4)
                

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=60)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--cfg', help='Config', type=click.Choice(['FFHQ', 'Cats']), required=False, metavar='STR', default='FFHQ', show_default=True)
@click.option('--image_mode', help='Image mode', type=click.Choice(['image', 'image_depth', 'image_raw']), required=False, metavar='STR', default='image', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float, help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--shapes', type=bool, help='Gen shapes for shape interpolation', default=False, show_default=True)
@click.option('--interpolate', type=bool, help='Interpolate between seeds', default=True, show_default=True)
@click.option('--ifstyle', type=bool, help='Interpolate between seeds', default=False, show_default=True)

def generate_images(
    network_pkl: str,
    seeds: List[int],
    shuffle_seed: Optional[int],
    truncation_psi: float,
    truncation_cutoff: int,
    grid: Tuple[int,int],
    num_keyframes: Optional[int],
    w_frames: int,
    outdir: str,
    reload_modules: bool,
    cfg: str,
    image_mode: str,
    sampling_multiplier: float,
    nrr: Optional[int],
    shapes: bool,
    interpolate: bool,
    ifstyle: bool,
):
    """Render a latent vector interpolation video.

    Examples:

    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')

    with torch.no_grad():
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    #     fine_tuned_model = torch.load(f'examples/PTI_results/small_nose/000300.pt')["g_ema"]#.to(device)

    #     G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).to(device)
    #     model_dict = G_new.state_dict()

    #     for k, v in fine_tuned_model.items():
    #             if k in model_dict.keys() and v.size() == model_dict[k].size():
    #                 model_dict[k] = v
    #             else:
    #                 print(k)
    #     for k, v in model_dict.items():
    #             if k in fine_tuned_model.keys() and v.size() == fine_tuned_model[k].size():
    #                 model_dict[k] = fine_tuned_model[k]
    #             else:
    #                 print(k)

    # G_new.load_state_dict(model_dict)
    # G_new.neural_rendering_resolution = G.neural_rendering_resolution
    # G_new.rendering_kwargs = G.rendering_kwargs
    # G = copy.deepcopy(G_new).eval().requires_grad_(False).to(device)


    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if nrr is not None: G.neural_rendering_resolution = nrr

    if truncation_cutoff == 0:
        truncation_psi = 1.0 # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14 # no truncation so doesn't matter where we cutoff

    if interpolate:
        gen_interp_video(G=G, outdir=outdir, bitrate='10M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds, shuffle_seed=shuffle_seed, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, gen_shapes=shapes, ifstyle=ifstyle)
    else:
        for seed in seeds:
            output = os.path.join(outdir, f'{seed}.mp4')
            seeds_ = [seed]
            gen_interp_video(G=G, outdir=outdir, bitrate='10M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds_, shuffle_seed=shuffle_seed, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, ifstyle=ifstyle)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
