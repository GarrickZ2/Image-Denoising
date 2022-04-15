import os
import cv2
import glob
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class SIDDDataset(Dataset):
    """
    https://www.eecs.yorku.ca/~kamel/sidd/index.php
    """
    CLEAN_IMG_NAME = "GT_SRGB_010.PNG"
    NOISY_IMG_NAME = "NOISY_SRGB_010.PNG"

    def __init__(self, root_dir, transform=None, data_type='train'):
        self.root_dir = root_dir
        self.dirs = pd.Series(glob.glob(f"{root_dir}/Data/**"))
        if data_type == 'train':
            self.dirs = self.dirs[:144]
        elif data_type == 'val':
            self.dirs = self.dirs[-16:]
        self.transform = transform
        self.data_type = data_type
        
    def __len__(self):
        return len(self.dirs)
    
    def __getitem__(self, idx):
        if self.data_type == 'val':
            idx += 144
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dir_name = self.dirs[idx]
        clean = cv2.imread(os.path.join(dir_name, self.CLEAN_IMG_NAME), cv2.IMREAD_GRAYSCALE)
        noisy = cv2.imread(os.path.join(dir_name, self.NOISY_IMG_NAME), cv2.IMREAD_GRAYSCALE)
        data = {'clean': clean, 'noisy': noisy}
        if self.transform:
            data = self.transform(data)

        return data

