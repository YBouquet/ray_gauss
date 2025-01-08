import logging
logging.basicConfig(
    filename='log_output.txt',  # Chemin du fichier
    level=logging.INFO,        # Niveau minimal de journalisation
    format='%(asctime)s - %(levelname)s - %(message)s'  # Format des messages
)
import argparse, os
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from classes import point_cloud,scene, cameras
from scripts import test
from utils.loss_utils import ssim
from utils.metrics_utils import PSNR
from lpipsPyTorch import lpips
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import matplotlib
import numpy as np

from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def focal2fov(focal, pixels):
    import math
    return 2*math.atan(pixels/(2*focal))

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r).permute(0,2,1)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def get_inputs(num_points=8):
    length = 0.5
    x = np.linspace(-1, 1, num_points) * length
    y = np.linspace(-1, 1, num_points) * length
    x, y = np.meshgrid(x, y)
    means3D = torch.from_numpy(np.stack([x,y, 0 * np.random.rand(*x.shape)], axis=-1).reshape(-1,3)).cuda().float()
    quats = torch.zeros(1,4).repeat(len(means3D), 1).cuda()
    quats[..., 0] = 1.
    scale = length /(num_points-1)
    scales = torch.zeros(1,3).repeat(len(means3D), 1).fill_(0.05).cuda()
    scales = torch.log(scales)
    return means3D, scales, quats

def get_cameras():
    intrins = torch.tensor([[711.1111,   0.0000, 256.0000,   0.0000],
               [  0.0000, 711.1111, 256.0000,   0.0000],
               [  0.0000,   0.0000,   1.0000,   0.0000],
               [  0.0000,   0.0000,   0.0000,   1.0000]]).cuda()
    c2w = torch.tensor([[-8.6086e-01,  3.7950e-01, -3.3896e-01,  0],
         [ 5.0884e-01,  6.4205e-01, -5.7346e-01,  1.1469e+00],
         [ 1.0934e-08, -6.6614e-01, -7.4583e-01,  1.4917e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]).cuda()
    c2w = torch.tensor([
        [1,  0, 0,  0],
        [ 0,  1, 0,  0],
        [ 0, 0, -1 ,  3],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]).cuda()
    width, height = 512, 512
    focal_x, focal_y = intrins[0, 0], intrins[1, 1]
    FoVx = focal2fov(focal_x, width)
    FoVy = focal2fov(focal_y, height)
    return c2w, FoVx, FoVy, height, width

if __name__ == "__main__":
    ############################################################################################################
    # load parameters
    
    means3D, scales, quats = get_inputs()
    viewmat, FoVx, FoVy, height, width = get_cameras()
    random.seed(0)
    np.random.seed(0)
    colors = matplotlib.colormaps['Accent'](np.random.randint(1,64, 64)/64)[..., :3]
    colors = torch.from_numpy(colors).cuda()
    opacity = torch.ones_like(means3D[:,0])

    fake_img = torch.zeros(1, height, width, 3).cuda()
    # create camera
    cameraInfo = cameras.Camera(
        colmap_id=0,
        R=viewmat[:3,:3].cpu().numpy(),
        T=viewmat[:3,3].cpu().numpy(),
        FoVy=FoVy,
        FoVx=FoVx,
        image=fake_img,
        uid = 0,
        gt_alpha_mask=fake_img,
        image_name="test"
    )
    cameraInfo.world_view_transform = torch.linalg.inv(torch.tensor(viewmat)).cuda()
    
    pcd = point_cloud.PointCloud(data_type="float32", device=device)
    pcd.positions = torch.tensor([[0,0,0]]).to(device)
    pcd.scales = scales[:1].to(device)
    pcd.quaternions = quats[:1].to(device)
    pcd.densities = opacity[:1].view(-1).to(device)
    pcd.rgb = colors[:1].to(device)
    pcd.spherical_harmonics = torch.zeros((len(means3D), 3, 16))[:1].to(device)
    pcd.lobe_axis = torch.ones((len(means3D),1,3))[:1].to(device)
    pcd.sph_gauss_features = torch.zeros((len(means3D),1,3))[:1].to(device)
    pcd.bandwidth_sharpness = torch.zeros((len(means3D),1))[:1].to(device)
    pcd.num_sph_gauss = 0
    
    [img] = test.render(
        pcd, [cameraInfo], 1024, 0,(1,1), False
    )
    
    plt.imshow(img)
    plt.show()
    