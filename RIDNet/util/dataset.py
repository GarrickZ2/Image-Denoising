import torch
import copy
import glob
import cv2
import numpy as np
import copy

class Denoising_dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, train_val, transform):
        super(Denoising_dataset, self).__init__()

        self.img_dir = [f for f in glob.glob(img_dir+'/**/*.jpg', recursive=True)]
        self.train_val = train_val
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_dir = self.img_dir[idx]

        clean = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)   
        noisy = np.copy(clean)         
        origin_img = copy.deepcopy(clean)
        
        noisy = self.gaussian_noise(clean)
        
        data = {'noisy': noisy, 'clean': clean}

        if self.transform:
            data = self.transform(data)

        return data
      
    def gaussian_noise(self, img, noise_level=[15, 25, 50]):
        sigma = np.random.choice(noise_level)
        gaussian_noise = np.random.normal(0, sigma, (img.shape[0], img.shape[1]))        
        
        noisy_img = img + gaussian_noise        
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        return noisy_img
    
