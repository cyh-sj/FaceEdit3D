from models.encoders.psp_encoders import GradualStyleEncoder
import torch
import dnnlib
import legacy
from inversion_samples import load_image, load_parameter
from camera_utils import FOV_to_intrinsics, LookAtPoseSampler
import numpy as np
import PIL.Image
import torch.nn.functional as F
from tqdm import tqdm
from training.triplane import TriPlaneGenerator
from torch_utils import misc
from lpips import LPIPS
from models.networks import define_mlp
from PIL import Image
import torchvision.transforms as transforms
from models.id_loss import IDLoss
import os
import mrcfile
import argparse

TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


def index_to_name(index):
    if index < 10:
        name = '0000' + str(index)
    elif index < 100:
        name = '000' + str(index)
    elif index < 1000:
        name = '00' + str(index)
    elif index < 10000:
        name = '0' + str(index)
    else:
        name = str(index)
    return name


def my_acti(w, type='sin'):
    if type == 'sin':
        return 0.5 * torch.sin(torch.tensor(np.pi) * (w - 0.5)) + 0.5
    elif type == 'sig':
        return 1 / (1 + torch.exp(-10 * (w - 0.5)))
    else:
        return w


def gen_mask(p):
    if p is None:
        mask_rect = torch.zeros([1, 512, 512])
        num = 75
        for i in range(35 * 2 + num, 223 * 2 - num):
            for j in range(32 * 2 + num, 220 * 2 - num):
                mask_rect[0][i][j] += 1
        return mask_rect
    index = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
    mask = torch.zeros([1, 512, 512])
    for i in index:
        mask += p == i
    mask_rect = torch.zeros([1, 512, 512])
    for i in range(35 * 2 + 50, 223 * 2):
        for j in range(32 * 2, 220 * 2):
            mask_rect[0][i][j] += 1
    return mask * mask_rect


def configure_optimizers(networks, lr=3e-4):
    params = list(networks.backbone.parameters()) + list(networks.renderer.parameters()) + list(networks.decoder.parameters())
    optimizer = torch.optim.Adam([{'params': params}], lr=lr)
    return optimizer


class FaceSwapCoach:
    def __init__(self):
        self.device = torch.device('cuda')

        self.l2 = torch.nn.MSELoss(reduction='mean')
        self.lpips = LPIPS(net='alex').to(self.device).eval()
        self.id_loss = IDLoss().to(self.device).eval()
        self.encoder = self.load_encoder()
        self.MLPs = self.load_mlps()
        self.decoder = self.load_decoder()

        self.gen_w_avg()

    def load_encoder(self):
        encoder = GradualStyleEncoder(50, 'ir_se')
        encoder_ckpt = torch.load('checkpoints/iteration_1000000.pt')
        encoder.load_state_dict(get_keys(encoder_ckpt, 'encoder'), strict=True)
        return encoder

    def load_mlps(self):
        MLPs = []
        mlp_ckpt = torch.load('checkpoints/iteration_100000.pt')
        for i in range(5):
            mlp = define_mlp(4)
            mlp.load_state_dict(get_keys(mlp_ckpt, f'MLP{i}'), strict=True)
            MLPs.append(mlp)
        return MLPs

    def load_decoder(self):
        network_pkl = 'checkpoints/ffhq512-128.pkl'
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(self.device)
            print("Reloading Modules!")
            G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(self.device)
            misc.copy_params_and_buffers(G, G_new, require_all=True)
            G_new.neural_rendering_resolution = G.neural_rendering_resolution
            G_new.rendering_kwargs = G.rendering_kwargs
            G = G_new.requires_grad_(True)
        return G

    def gen_w_avg(self):
        intrinsics = FOV_to_intrinsics(18.837, device=self.device)
        cam_pivot = torch.tensor(self.decoder.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=self.device)
        cam_radius = self.decoder.rendering_kwargs.get('avg_camera_radius', 2.7)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius,
                                                               device=self.device)
        constant_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        num_samples = 10000
        z_samples = np.random.RandomState(123).randn(num_samples, 512)
        w_samples = self.decoder.mapping(torch.from_numpy(z_samples).to(self.device),
                                         constant_params.repeat([num_samples, 1]), truncation_psi=0.7,
                                         truncation_cutoff=14)
        w_samples = w_samples[:, :1, :].cpu().detach().numpy().astype(np.float32)
        w_avg = np.mean(w_samples, axis=0, keepdims=True)
        w_avg = np.repeat(w_avg, 14, axis=1)
        self.w_avg = torch.tensor(w_avg).to(self.device)

    def inversion(self, x, y):
        with torch.no_grad():
            x_ws = self.encoder(x.cpu()).to(self.device) + self.w_avg
            y_ws = self.encoder(y.cpu()).to(self.device) + self.w_avg

        return x_ws, y_ws

    def latent_interpolation(self, x_ws, y_ws):
        start_layer = 5

        x_codes, y_codes = [], []
        for i in range(start_layer, start_layer + 5):
            x_codes.append(x_ws[:, i: i + 1])
            y_codes.append(y_ws[:, i: i + 1])

        yhat_codes = [y_ws[:, :start_layer]]
        for i in range(start_layer, start_layer + 5):
            i = i - start_layer
            MLP = self.MLPs[i]
            rho = MLP(torch.cat([x_codes[i], y_codes[i]], dim=2))
            rho = my_acti(rho, type='sig')
            yhat_codes.append(y_codes[i] * rho + x_codes[i] * (1 - rho))

        yhat_codes.append(y_ws[:, start_layer + 5:])
        ws = torch.cat(yhat_codes, dim=1)

        return ws

    def synthesis(self, x_ws, ws, in_cp, out_cp):
        x_rec = self.decoder.synthesis(x_ws, in_cp)['image']
        y_hat = self.decoder.synthesis(ws, out_cp)['image']
        y_hat_ = self.decoder.synthesis(ws, in_cp)['image']

        return x_rec, y_hat, y_hat_

    def cal_loss(self, x, y, x_rec, y_hat, y_hat_, mask, tune):
        loss = 0.0
        stop_training = False

        theta_1 = 1.0
        theta_2 = 1.0

        x_rec = F.interpolate(x_rec, size=[256, 256], mode='bilinear', align_corners=True)
        y_hat = F.interpolate(y_hat, size=[256, 256], mode='bilinear', align_corners=True)

        loss_l2 = self.l2(x, x_rec) * theta_1
        if tune:
            loss_l2 += self.l2(y * (1 - mask), y_hat * (1 - mask)) * theta_2

        loss_lpips = self.lpips(x, x_rec) * theta_1
        if tune:
            loss_lpips += self.lpips(y * (1 - mask), y_hat * (1 - mask)) * theta_2
        loss_lpips = torch.squeeze(loss_lpips)

        loss_id = self.id_loss.forward_se(x, x_rec)
        if tune:
            loss_id += self.id_loss.forward_se(x, y_hat) + self.id_loss.forward_se(x, y_hat_)

        loss += loss_l2 * 1.0 + loss_lpips * 1.0 + loss_id * 5.0

        if tune and loss_l2 < 0.02:
            stop_training = True

        return loss, stop_training

    def train(self, args):
        in_name, out_name = index_to_name(args.from_index), index_to_name(args.to_index)

        in_name = 'zcg'
        out_name = 'dc'
        in_image = load_image(args.dataroot + 'final_crops/' + in_name + '.jpg', self.device)
        out_image = load_image(args.dataroot + 'final_crops/' + out_name + '.jpg', self.device)

        name = in_name + '_' + out_name

        """ Data Preparing """
        # in_image = load_image(args.dataroot + 'final_crops/' + in_name + '.png', self.device)
        # out_image = load_image(args.dataroot + 'final_crops/' + out_name + '.png', self.device)

        in_cp = load_parameter(args.dataroot + 'camera_pose/' + in_name + '.npy', self.device)
        out_cp = load_parameter(args.dataroot + 'camera_pose/' + out_name + '.npy', self.device)

        in_img = (in_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        out_img = (out_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        imgs = [in_img, out_img]

        use_label = False
        if use_label:
            out_label_path = '/2t/datasets/EG3D/labels/' + out_name + '.png'
            out_label = Image.open(out_label_path).convert('L')
            out_label_tensor = TRANSFORM(out_label) * 255.0
            out_face_mask = gen_mask(out_label_tensor)
            out_face_mask = F.interpolate(out_face_mask.unsqueeze(0), size=[256, 256], mode='bilinear',
                                          align_corners=True).cuda()
            mask = out_face_mask
        else:
            face_mask = gen_mask(None)
            face_mask = F.interpolate(face_mask.unsqueeze(0), size=[256, 256], mode='bilinear',
                                      align_corners=True).cuda()
            mask = face_mask

        """ Face Swap """
        x = F.interpolate(in_image, size=[256, 256], mode='bilinear', align_corners=True)
        y = F.interpolate(out_image, size=[256, 256], mode='bilinear', align_corners=True)
        x_ws, y_ws = self.inversion(x, y)

        """ Optimize G """
        self.decoder.train()
        optimizer = configure_optimizers(self.decoder, args.lr)
        ws = None

        for step in tqdm(range(args.epoch)):
            ws = self.latent_interpolation(x_ws, y_ws)
            x_rec, y_hat, y_hat_ = self.synthesis(x_ws, ws, in_cp, out_cp)

            start_tuning_epoch = int(0.5 * args.epoch)
            tuning_epochs = args.epoch - start_tuning_epoch

            loss, stop_training = self.cal_loss(x, y, x_rec, y_hat, y_hat_, mask, step >= start_tuning_epoch)

            optimizer.zero_grad()
            if stop_training:
                break
            loss.backward()
            optimizer.step()

            if step >= int(start_tuning_epoch):
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = old_lr - args.lr / tuning_epochs
                    param_group['lr'] = new_lr

        torch.save(self.decoder, f'latents/model_encoder_' + name + '.pt')

        """ Output """
        yhat_out = self.decoder.synthesis(ws, out_cp)['image']
        yhat_in = self.decoder.synthesis(ws, in_cp)['image']
        id_sim = [1 - self.id_loss.forward_se(x, F.interpolate(yhat_out, size=[256, 256], mode='bilinear',
                                                               align_corners=True)).item(),
                  1 - self.id_loss.forward_se(x, F.interpolate(yhat_in, size=[256, 256], mode='bilinear',
                                                               align_corners=True)).item()]
        yhat_out = (yhat_out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        yhat_in = (yhat_in.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        with open('faceswap/results.txt', 'a') as f:
            f.write(str(args.from_index) + '_' + str(args.to_index) + ' ' + str(id_sim[0]) + ' ' + str(id_sim[1]))
            f.write('\n')

        print(id_sim)

        imgs.append(yhat_out)
        imgs.append(yhat_in)

        img = torch.cat(imgs, dim=2)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'faceswap/' + name + '.png')
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'faceswap/latest.png')
