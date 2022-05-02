import glob
import pandas as pd
import torch
import os
import torchvision.transforms as transforms
import torch.distributions as dist
import tqdm
from PIL import Image
from torch.utils.data import Dataset
import multiprocessing as mp
import argparse

default_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def denormalize(tensors, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
  """ Denormalizes image tensors using mean and std """
  for c in range(3):
      tensors[:, c].mul_(std[c]).add_(mean[c])
  return torch.clamp(tensors, 0, 255)

class AdditiveGaussianWhiteNoise(object):
    def __init__(self, mean=0., std=25.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + (torch.randn(tensor.size()) * self.std + self.mean) / 255.

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


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
                 noise_generator=fake_noise_model,
                 data_type='train',
                 parallel=False,
                 machine_num=2,
                 machine_id=0,
                 skip_num=0,
                 load_fake=False,
                 limit=None):
        self.root_dir = root_dir
        self.data_type = data_type
        self.input_dirs = pd.Series(glob.glob(f"{root_dir}/{data_type}/SIDD/input_crops/**"))
        self.target_dirs = pd.Series(glob.glob(f"{root_dir}/{data_type}/SIDD/target_crops/**"))
        self.fake_dirs = None
        self.skip_num = skip_num
        if load_fake:
            self.fake_dirs = pd.Series(glob.glob(f"{root_dir}/{data_type}/SIDD/noisy_crops/**"))
        self.transform = transform
        self.noise_generator = noise_generator
        self.parallel = parallel
        self.machine_num = machine_num
        self.machine_id = machine_id
        if limit is not None:
            self.input_dirs = self.input_dirs[:limit]
            self.target_dirs = self.target_dirs[:limit]
        self.length = len(self.input_dirs)
        if self.parallel:
            self.length = int(len(self.input_dirs) / self.machine_num)

    def __len__(self):
        return self.length

    def __getitem__(self, origin_id):
        if self.parallel:
            idx = origin_id * self.machine_num + self.machine_id
        else:
            idx = origin_id
        if idx < self.skip_num:
            return None, None, None
        else:
            self.skip_num = 0
        if self.fake_dirs is not None:
            return torch.load(self.fake_dirs[idx])

        clean = Image.open(self.target_dirs[idx])
        true_noisy = Image.open(self.input_dirs[idx])

        if self.transform:
            state = torch.get_rng_state()
            clean = self.transform(clean)
            torch.set_rng_state(state)
            true_noisy = self.transform(true_noisy)

        fake_noisy = self.noise_generator(clean)

        return clean, true_noisy, fake_noisy

    def __process_fake_noise_iamge(self, work_id, worker_num):
        process = tqdm.tqdm(range(int(len(self.input_dirs) / worker_num) + 1))
        for rounds in process:
            idx = rounds * worker_num + work_id
            if idx >= len(self.input_dirs):
                break
            process.set_description(f'Worker Id: {work_id}')
            filename = self.target_dirs[idx]
            dirs = filename.split('/')
            dirs[-1] = dirs[-1].split('.')[0] + '.pt'
            dirs[-2] = 'noisy_crops'
            target_filename = "/".join(dirs)
            clean = Image.open(filename)
            true_noisy = Image.open(self.input_dirs[idx])
            if self.transform:
                state = torch.get_rng_state()
                clean = self.transform(clean)
                torch.set_rng_state(state)
                true_noisy = self.transform(true_noisy)

            fake_noisy = self.noise_generator(clean)
            data = (clean, true_noisy, fake_noisy)
            torch.save(data, target_filename)
        process.set_description(f'Worker {work_id} Finished Job')
        process.close()

    def generate_noise_image(self, workers):
        print(f'Start Processing with {workers} workers')
        if not os.path.exists(f'{self.root_dir}/{self.data_type}/SIDD/noisy_crops/'):
            os.mkdir(f'{self.root_dir}/{self.data_type}/SIDD/noisy_crops/')
        process_pool = []
        for i in range(workers):
            proc = mp.Process(target=self.__process_fake_noise_iamge, args=(i, workers,))
            process_pool.append(proc)
            proc.start()
        for each in process_pool:
            each.join()
        print('Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./Datasets', help='root dir')
    parser.add_argument('--workers', type=int, default=4, help='How many workers work together')
    args = parser.parse_args(args=[])

    train_ds = SIDDSmallDataset(args.root, noise_generator=AdditiveGaussianWhiteNoise())
    print('Start To process for Train Dataset')
    train_ds.generate_noise_image(args.workers)
    val_ds = SIDDSmallDataset(args.root, data_type='val', noise_generator=AdditiveGaussianWhiteNoise())
    print('Start To process for Val Dataset')
    val_ds.generate_noise_image(args.workers)
