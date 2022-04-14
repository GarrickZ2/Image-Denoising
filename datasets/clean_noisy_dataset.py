import os
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

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.dirs = pd.Series(glob.glob(f"{root_dir}/Data/**"))
        self.transform = transform
        
    def __len__(self):
        return len(self.dirs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dir_name = self.dirs[idx]
        clean = Image.open(os.path.join(dir_name, self.CLEAN_IMG_NAME))
        noisy = Image.open(os.path.join(dir_name, self.NOISY_IMG_NAME))
        
        if self.transform:
            state = torch.get_rng_state()
            clean = self.transform(clean)
            torch.set_rng_state(state)
            noisy = self.transform(noisy)

        return {'clean': clean, 'noisy': noisy}
