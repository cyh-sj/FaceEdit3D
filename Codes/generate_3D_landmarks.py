import os
import re
import math

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
import legacy
import argparse
from PIL import Image
from camera_utils import LookAtPoseSampler
from torch_utils import misc
from training.triplane import TriPlaneGenerator
import torchvision.transforms as transforms

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe.python._framework_bindings import image as mp_image
from mediapipe.python._framework_bindings import image_frame as mp_image_frame

from wrap_2D.controller_warp_2D import WarpController

from multiprocessing import Process
from torch.nn import functional as F

TRANSFORM = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

def load_image(image_path, device):
    image = Image.open(image_path).convert('RGB')
    image = TRANSFORM(image).unsqueeze(0)
    return image.to(device)



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


def gen_face_from_w(G, angle, w, seed, camera_lookat_point, radius, device):
    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + 0, 3.14/2 - 3.14 / 8, camera_lookat_point, radius=radius, device=device)
    intrinsics = torch.tensor([[4.4652 , 0, 0.5], [0, 4.4652 , 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c[0:1]

    if w is None:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        w = G.mapping(z, c, truncation_psi=0.7, truncation_cutoff=14)

    triplanes = G.backbone.synthesis(ws=w)
    triplanes = triplanes.view(len(triplanes), 3, 32, triplanes.shape[-2], triplanes.shape[-1])

    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + angle[0], 3.14/2 + angle[1] ,camera_lookat_point, radius=radius, device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c[0:1]
    cam2world_matrix = cam2world_pose.view(-1, 4, 4)
    intrinsics = c[:, 16:25].view(-1, 3, 3)

    ray_origins, ray_directions = G.ray_sampler(cam2world_matrix, intrinsics, G.neural_rendering_resolution)
    N, M, _ = ray_origins.shape
    
    feature_samples, depth_samples, _ = G.renderer(triplanes, G.decoder, ray_origins, ray_directions, G.rendering_kwargs)
    H = W = G.neural_rendering_resolution

    feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
    rgb_image = feature_image[:, :3]
    sr_image = G.superresolution(rgb_image, feature_image, w)
    depth_img = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

    depth_img = - depth_img.permute(0, 2, 3, 1)
    depth_img = (((depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())) * 255).clamp(0, 255).to(torch.uint8)

    img = (sr_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img[0].cpu().numpy()
    
    return img, depth_img, triplanes, w


def gen_face_from_triplane(G, angle, w, triplane, camera_lookat_point, radius, device):
    intrinsics = torch.tensor([[4.4652 , 0, 0.5], [0, 4.4652 , 0.5], [0, 0, 1]], device=device)

    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + angle[0], 3.14/2 + angle[1] ,camera_lookat_point, radius=radius, device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c[0:1]
    cam2world_matrix = cam2world_pose.view(-1, 4, 4)
    intrinsics = c[:, 16:25].view(-1, 3, 3)

    ray_origins, ray_directions = G.ray_sampler(cam2world_matrix, intrinsics, G.neural_rendering_resolution)
    N, M, _ = ray_origins.shape
    
    feature_samples, depth_samples, _ = G.renderer(triplanes, G.decoder, ray_origins, ray_directions, G.rendering_kwargs)
    H = W = G.neural_rendering_resolution

    feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
    rgb_image = feature_image[:, :3]
    sr_image = G.superresolution(rgb_image, feature_image, w)
    depth_img = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

    depth_img = - depth_img.permute(0, 2, 3, 1)
    depth_img = (((depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())) * 255).clamp(0, 255).to(torch.uint8)
    
    return sr_image, depth_img


def gen_3D_coors(G, triplane, camera_lookat_point, radius, device):
    intrinsics = torch.tensor([[4.4652 , 0, 0.5], [0, 4.4652 , 0.5], [0, 0, 1]], device=device)

    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + 0, 3.14/2 + 0, camera_lookat_point, radius=radius, device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c[0:1]
    cam2world_matrix = cam2world_pose.view(-1, 4, 4)
    intrinsics = c[:, 16:25].view(-1, 3, 3)

    ray_origins, ray_directions = G.ray_sampler(cam2world_matrix, intrinsics, G.neural_rendering_resolution)
    
    out_sigma, out_cor = G.renderer.gen_3D_coor(triplanes, G.decoder, ray_origins, ray_directions, G.rendering_kwargs)
    
    return out_sigma, out_cor


def gen_3D_ldms(out_sigma, out_cor, ldm_2ds, device):
    ldm_3d = None
    for ldm in ldm_2ds:
        y = int(float(ldm[0]) * 128)
        x = int(float(ldm[1]) * 128)
        sigmas_line = out_sigma[0,y,x,:]
        idx = torch.argmax(sigmas_line)
        coo = out_cor[0,y,x,idx].unsqueeze(0)
        if ldm_3d is None:
            ldm_3d = coo + 0.5
        else:
            ldm_3d = torch.cat([ldm_3d, coo + 0.5], dim = 0)

    ldm_edge = torch.tensor([[0.2, 0.2, 0.6], [0.2, 0.8, 0.9], [0.8, 0.2, 0.9], [0.8, 0.8, 0.6]], device=device)
    ldm_3d = torch.cat([ldm_3d, ldm_edge], dim = 0)
    
    return ldm_3d


def detec_2D_ldms(dector, image):
    rgb_image = mp_image.Image(image_format=mp_image_frame.ImageFormat.SRGB, data = np.asarray(PIL.Image.fromarray(image, "RGB")))
    detection_result = detector.detect(rgb_image)
    landmarks_2D = []
    face_landmarks_list = detection_result.face_landmarks[0]
    for idx in [10, 199, 19, 4, 6, 64, 279, 0, 17, 57, 306, 25,26,145,159, 55, 52,46, 50, 285, 282, 276, 263, 362, 374, 385, 280, 132, 361]:
    # for idx in range(len(face_landmarks_list)):
        landmark = face_landmarks_list[idx]
        ldm = []
        ldm.append(landmark.y)
        ldm.append(landmark.x)
        landmarks_2D.append(ldm)
    landmarks_2D = np.array(landmarks_2D)

    return landmarks_2D
        

def save_images(img, depth_image, outdir, image_name, ldks_tgt_3d=None, ldks_src_3d=None):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if ldks_tgt_3d is not None:
        for point in ldks_tgt_3d[:,[0,1]]:
            cv2.circle(img, (int(512 * point[1]), int(512 * (point[0]))), 1, color=(0, 0, 255), thickness=2)
    
    if ldks_src_3d is not None: 
        for point in ldks_src_3d[:,[0,1]]:
            cv2.circle(img, (int((point[1]-0.2)/0.6*512*math.cos(angle)), int(512 * (1 - (point[0] - 1/6) * 0.6 / 0.5))), 1, color=(255, 0, 0), thickness=2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/{image_name}.png')
    if depth_image is not None:
        PIL.Image.fromarray(depth_img[0,:,:,0].cpu().numpy(), 'L').save(f'{outdir}/{image_name}_depth.png')


def get_parser():
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--seeds', type=int, help='seeds number', default=0)
    parser.add_argument("--names", type=str, help='path to data directory', default='test')

    return parser


if __name__ == "__main__":
    # set mediapipe
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    device = torch.device('cuda')
    args = get_parser().parse_args()

    # set generator
    with torch.no_grad():
        with dnnlib.util.open_url('ffhqrebalanced512-128.pkl') as f:
        # with dnnlib.util.open_url('afhqcats512-128.pkl') as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * 2)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * 2)
    G.rendering_kwargs['ray_start']= 2.25
    G.rendering_kwargs['ray_end'] = 3.3

    camera_lookat_point = torch.tensor([0, 0, 0.2], device=device)
    intrinsics = torch.tensor([[4.4652 , 0, 0.5], [0, 4.4652 , 0.5], [0, 0, 1]], device=device)

    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + 0, 3.14/2 + 0, camera_lookat_point, radius=2.7, device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c[0:1]
    if True:
        num_samples = 10000
        z_samples = np.random.RandomState(123).randn(num_samples, 512)
        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c.repeat([num_samples, 1]), truncation_psi=0.7, truncation_cutoff=14)
        w_samples = w_samples[:, :1, :].cpu().detach().numpy().astype(np.float32)
        w_avg = np.mean(w_samples, axis=0, keepdims=True)

        w_avg = np.repeat(w_avg, 14, axis=1)
        w_avg = torch.tensor(w_avg).to(device)


    image, _, triplanes, w = gen_face_from_w(G, [0, 0], None, args.seeds, camera_lookat_point, 2.7, device)


    ldm_2d = detec_2D_ldms(detector, image)

    image, _ = gen_face_from_triplane(G, [0, 0], w, triplanes, camera_lookat_point, 2.7, device)

    out_sigma, out_coo = gen_3D_coors(G, triplanes, camera_lookat_point, 2.7, device)
    ldm_3d = gen_3D_ldms(out_sigma, out_coo, ldm_2d, device)

    with open('source_3d_ldms.txt', 'w') as f:
        for coo in ldm_3d:
            f.write("{:.4f}".format(coo[1]) + ' ' + "{:.4f}".format(coo[0]) + ' '  + "{:.4f}".format(coo[2]) + '\n')

    with open('target_3d_ldms.txt', 'w') as f:
        for coo in ldm_3d:
            f.write("{:.4f}".format(coo[1]) + ' ' + "{:.4f}".format(coo[0]) + ' '  + "{:.4f}".format(coo[2]) + '\n')

    img = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img[0].cpu().numpy()

    
    # save_images(image, None, 'out', 'test', ldm_2d)
    PIL.Image.fromarray(img, 'RGB').save(f'generated_img.png')
    torch.save(w, f'generated_w.pt')