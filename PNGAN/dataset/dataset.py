import glob
import pandas as pd
import torch
import torchvision.transforms as transforms
import torch.distributions as dist
from PIL import Image
from torch.utils.data import Dataset

default_transform = transforms.Compose([
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
        return torch.clamp(image + noise, 0, 1)

    return _func


fake_noise_model = add_noise()


class SIDDSmallDataset(Dataset):
    def __init__(self,
                 root_dir,
                 transform=default_transform,
                 fake_preserve=True,
                 noise_generator=fake_noise_model,
                 data_type='train',
                 parallel=False,
                 machine_num=2,
                 machine_id=0,
                 limit=None):
        self.root_dir = root_dir
        self.input_dirs = pd.Series(glob.glob(f"{root_dir}/{data_type}/SIDD/input_crops/**"))
        self.target_dirs = pd.Series(glob.glob(f"{root_dir}/{data_type}/SIDD/target_crops/**"))
        self.transform = transform
        self.preserve = fake_preserve
        self.noise_generator = noise_generator
        self.fake_image_set = {}
        self.parallel = parallel
        self.machine_num = machine_num
        self.machine_id = machine_id
        if limit is not None:
            self.input_dirs = self.input_dirs[:limit]
            self.target_dirs = self.target_dirs[:limit]

    def __len__(self):
        if self.parallel:
            return len(self.input_dirs) / self.machine_num
        return len(self.input_dirs)

    def __getitem__(self, origin_id):
        if self.parallel:
            idx = origin_id * self.machine_num + self.machine_id
        else:
            idx = origin_id

        clean = Image.open(self.target_dirs[idx])
        true_noisy = Image.open(self.input_dirs[idx])

        if self.transform:
            state = torch.get_rng_state()
            clean = self.transform(clean)
            torch.set_rng_state(state)
            true_noisy = self.transform(true_noisy)

        if self.preserve:
            if idx in self.fake_image_set.keys():
                fake_noisy = self.fake_image_set[idx]
            else:
                fake_noisy = self.noise_generator(clean)
                self.fake_image_set[idx] = fake_noisy
        else:
            fake_noisy = self.noise_generator(clean)

        return clean, true_noisy, fake_noisy
