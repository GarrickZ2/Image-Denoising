import os
import glob
import pandas as pd
import torch
import torchvision.transforms as transforms
import torch.distributions as dist
from PIL import Image
from torch.utils.data import Dataset, DataLoader

default_patch_size = (128, 128)
default_transform = transforms.Compose([
    transforms.RandomResizedCrop(default_patch_size),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
])


def random_noise_levels_sidd():
    """ Where read_noise in SIDD is not 0 """
    log_min_shot_noise = torch.log10(torch.Tensor([0.0001]))
    log_max_shot_noise = torch.log10(torch.Tensor([0.012]))
    distribution = dist.uniform.Uniform(log_min_shot_noise, log_max_shot_noise)

    log_shot_noise = distribution.sample()
    shot_noise = torch.pow(10, log_shot_noise)
    distribution = dist.normal.Normal(torch.Tensor([0.0]), torch.Tensor([0.26]))
    read_noise = distribution.sample()
    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + read_noise
    read_noise = torch.pow(10, log_read_noise)
    return shot_noise, read_noise


def add_noise(noise_func=random_noise_levels_sidd):
    """Adds random shot (proportional to image) and read (independent) noise."""

    def _func(image):
        shot_noise, read_noise = random_noise_levels_sidd()
        variance = image * shot_noise + read_noise
        mean = torch.Tensor([0.0])
        distribution = dist.normal.Normal(mean, torch.sqrt(variance))
        noise = distribution.sample()
        return image + noise

    return _func


fake_noise_model = add_noise()


class SIDDSmallDataset(Dataset):
    """
    https://www.eecs.yorku.ca/~kamel/sidd/index.php
    """
    CLEAN_IMG_NAME = "GT_SRGB_010.PNG"
    NOISY_IMG_NAME = "NOISY_SRGB_010.PNG"

    def __init__(self,
                 root_dir,
                 transform=default_transform,
                 data_type='train',
                 fake_noise_model=fake_noise_model):
        self.root_dir = root_dir
        self.dirs = pd.Series(glob.glob(f"{root_dir}/{data_type}/**"))
        self.transform = transform
        self.fake_noise_model = fake_noise_model

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dir_name = self.dirs[idx]
        clean = Image.open(os.path.join(dir_name, self.CLEAN_IMG_NAME))
        true_noisy = Image.open(os.path.join(dir_name, self.NOISY_IMG_NAME))

        if self.transform:
            state = torch.get_rng_state()
            clean = self.transform(clean)
            torch.set_rng_state(state)
            true_noisy = self.transform(true_noisy)

        fake_noisy = self.fake_noise_model(clean)

        return clean, true_noisy, fake_noisy
