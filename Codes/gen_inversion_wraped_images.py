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
import torch.nn.functional as F
import PIL.Image
import cv2
from PIL import Image

import legacy

from camera_utils import LookAtPoseSampler
from torch_utils import misc
from training.triplane import TriPlaneGenerator
import torchvision.transforms as transforms

from argparse import Namespace

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


TRANSFORM = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

def load_image(image_path, device):
    image = Image.open(image_path).convert('RGB')
    image = TRANSFORM(image).unsqueeze(0)
    return image.to(device)

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

#----------------------------------------------------------------------------

def gen_interp_video(G, outdir: str, seeds, shuffle_seed=None, w_frames=60*4, kind='cubic', grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, truncation_cutoff=14, cfg='FFHQ', image_mode='image', gen_shapes=False, device=torch.device('cuda'), edit_shape=False, edit_exp=False, direction=None, **video_kwargs):
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

    camera_lookat_point = torch.tensor([0, 0, 0.2], device=device) if cfg == 'FFHQ' else torch.tensor([0, 0, 0.2], device=device)

    zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.0], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(len(zs), 1)
    ws = G.mapping(z=zs, c=c, truncation_psi=psi, truncation_cutoff=truncation_cutoff)
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

    # Render video.
    voxel_resolution = 512


    pitch_range = 0
    yaw_range = 2 * 3.14

    radius = 2.7
    angles = [0.0, 3.14/6, -3.14/6]

    # ICCV 2023
    from configs.swin_config import get_config
    from goae import GOAEncoder

    swin_config = get_config()
    encoder = GOAEncoder(swin_config).to(device)
    E_ckpt = torch.load('encoder.pt', map_location=device)
    encoder.load_state_dict(E_ckpt)
    encoder.eval()
    encoder = encoder.to(device)
    
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        cam2world_pose = LookAtPoseSampler.sample(3.14/2 + 0, 3.14/2 + pitch_range ,camera_lookat_point, radius=radius, device=device)
        intrinsics = torch.tensor([[4.4652 , 0, 0.5], [0, 4.4652 , 0.5], [0, 0, 1]], device=device)
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        c = c[0:1]
        # w = G.mapping(z, c, truncation_psi=psi, truncation_cutoff=truncation_cutoff)
        w = torch.load('generated_w.pt', map_location=device).detach()

        num_samples = 10000
        z_samples = np.random.RandomState(123).randn(num_samples, 512)
        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c.repeat([num_samples, 1]), truncation_psi=0.7, truncation_cutoff=14)
        w_samples = w_samples[:, :1, :].cpu().detach().numpy().astype(np.float32)
        w_avg = np.mean(w_samples, axis=0, keepdims=True)

        w_avg = np.repeat(w_avg, 14, axis=1)
        w_avg = torch.tensor(w_avg).to(device)


        images = load_image(f'{outdir}/seed{seed:04d}_direction0.00.png', device)
        x = F.interpolate(images, size=[256, 256], mode='bilinear', align_corners=True)

        images2 = load_image(f'{outdir}/origin{seed:04d}_direction0.00.png', device)
        x2 = F.interpolate(images2, size=[256, 256], mode='bilinear', align_corners=True)

        with torch.no_grad():
            w_1 = encoder(x).cuda() + w_avg
            w_2 = encoder(x2).cuda()  + w_avg

        w_pivot = copy.deepcopy(w)

        if direction is not None:
            edit_d = 1.5 * torch.tensor(np.load(direction)).to(device)
        else:
            edit_d = 1.5 * (w_1 - w_2).to(device)

        if edit_shape:
            w_pivot[:, 3:5, :] += edit_d[:, 3:5, :]
        
        if edit_exp:
            w_pivot[:, 6:7, :] += edit_d[:, 6:7, :]

        triplanes_1 = G.backbone.synthesis(ws=w_pivot)
        triplanes_1 = triplanes_1.view(len(triplanes_1), 3, 32, triplanes_1.shape[-2], triplanes_1.shape[-1])
        planes = triplanes_1

        pitch_range = -0.05
        for angle in angles:
            cam2world_pose = LookAtPoseSampler.sample(3.14/2 + angle, 3.14/2 + pitch_range ,camera_lookat_point, radius=radius, device=device)
            intrinsics = torch.tensor([[4.4652 , 0, 0.5], [0, 4.4652 , 0.5], [0, 0, 1]], device=device)
            c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            c = c[0:1]

            cam2world_matrix = c[:, :16].view(-1, 4, 4)
            intrinsics = c[:, 16:25].view(-1, 3, 3)
            G.rendering_kwargs['ray_start'] = 2.85 + radius - 3.3
            G.rendering_kwargs['ray_end'] = 3.9 + radius - 3.3

            ray_origins, ray_directions = G.ray_sampler(cam2world_matrix, intrinsics, G.neural_rendering_resolution)
            N, M, _ = ray_origins.shape

            feature_samples, depth_samples, _ = G.renderer(planes, G.decoder, ray_origins, ray_directions, G.rendering_kwargs)
            H = W = G.neural_rendering_resolution

            feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
            rgb_image = feature_image[:, :3]

            sr_image = G.superresolution(rgb_image, feature_image, w)

            img = (sr_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = img[0].cpu().numpy()

            PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/inverwrap_seed{seed:04d}_direction{angle:.2f}.png')

        shape_res = 512
        if False:
            # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
            max_batch=1000000

            samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
            samples = samples.to(z.device)
            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
            transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
            transformed_ray_directions_expanded[..., -1] = -1

            # planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

            head = 0
            with tqdm(total = samples.shape[1]) as pbar:
                with torch.no_grad():
                    while head < samples.shape[1]:
                        torch.manual_seed(0)
                        sigma = G.renderer.run_model(planes, G.decoder, samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], G.rendering_kwargs)['sigma']  #G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
                        sigmas[:, head:head+max_batch] = sigma
                        head += max_batch
                        pbar.update(max_batch)

            sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
            sigmas = np.flip(sigmas, 0)

            # Trim the border of the extracted cube
            pad = int(30 * shape_res / 256)
            pad_value = -1000
            sigmas[:pad] = pad_value
            sigmas[-pad:] = pad_value
            sigmas[:, :pad] = pad_value
            sigmas[:, -pad:] = pad_value
            sigmas[:, :, :pad] = pad_value
            sigmas[:, :, -pad:] = pad_value

            # if shape_format == '.ply':
            from shape_utils import convert_sdf_samples_to_ply
            convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'seed{seed:04d}.ply'), level=10)
 

                

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
@click.option('--seeds', type=parse_range, help='List of random seeds', required=False, default=0)
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
@click.option('--edit_shape', type=bool, help='editing shapes', default=False, show_default=True)
@click.option('--edit_exp', type=bool, help='editing exppresions', default=False, show_default=True)
@click.option('--directions', help='saved directions', default=None)

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
    edit_shape,
    edit_exp,
    directions
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

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if nrr is not None: G.neural_rendering_resolution = nrr

    if truncation_cutoff == 0:
        truncation_psi = 1.0 # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14 # no truncation so doesn't matter where we cutoff

    if interpolate:
        gen_interp_video(G=G, outdir=outdir, bitrate='10M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds, shuffle_seed=shuffle_seed, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, gen_shapes=shapes, edit_shape=edit_shape, edit_exp=edit_exp, direction=directions)
    else:
        for seed in seeds:
            output = os.path.join(outdir, f'{seed}.mp4')
            seeds_ = [seed]
            gen_interp_video(G=G, outdir=outdir, bitrate='10M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds_, shuffle_seed=shuffle_seed, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, edit_shape=edit_shape, edit_exp=edit_exp, direction=directions)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
