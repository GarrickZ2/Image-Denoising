import os, shutil
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
import numpy as np


def download_pretrained_model():
    task = os.popen(
        'wget https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth -P '
        'Denoising/pretrained_models')
    task.close()


def download_demo_dataset():
    task = os.popen('rm -r demo/*')
    task.close()
    task = os.popen('wget https://github.com/swz30/Restormer/releases/download/v1.0/sample_images.zip -P demo')
    task.close()
    shutil.unpack_archive('demo/sample_images.zip', 'demo/')
    os.remove('demo/sample_images.zip')


# Get model weights and parameters
parameters = {'inp_channels': 3, 'out_channels': 3, 'dim': 48, 'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4,
              'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66, 'bias': False, 'LayerNorm_type': 'BiasFree',
              'dual_pixel_task': False}
weights = os.path.join('Denoising', 'pretrained_models', 'real_denoising.pth')

load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
model = load_arch['Restormer'](**parameters)
model.cuda()

checkpoint = torch.load(weights)
model.load_state_dict(checkpoint['params'])
model.eval()
