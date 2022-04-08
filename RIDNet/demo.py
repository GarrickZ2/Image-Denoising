import os
import torch
import cv2
import numpy as np

from model.RIDNet import RIDNet
from util.dataset import *


def gaussian_noise(img, noise_level=[5, 10, 15, 20, 25, 30]):
    sigma = np.random.choice(noise_level)
    gaussian_noise = np.random.normal(0, sigma, (img.shape[0], img.shape[1]))        
        
    noisy_img = img + gaussian_noise        
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)        
    return noisy_img


def demo():   
    model = RIDNet(in_channels=1, out_channels=1, num_feautres=32)   
    
    checkpoint = torch.load('./weight/weight.pth')
    model.load_state_dict(checkpoint['model_state_dict'])    
    criterion = checkpoint['loss']    
    
    if torch.cuda.is_available():    
        device = torch.device("cuda:0")
        print(device)
        model.to(device)
    
    img = cv2.imread(v, cv2.IMREAD_GRAYSCALE)
    origin_img = copy.deepcopy(img)
    
    img = gaussian_noise(img, noise_level=[15])    
    img = np.expand_dims(img, -1)
    img = img / 255.
    img = np.expand_dims(img , 0)
    img = torch.from_numpy(img).type(torch.float32)
    img = img.permute(0, 3, 1, 2)
    img = img.to(device)

    pred = model(img)
    output = pred[0].cpu().numpy().transpose(1, 2, 0)
    output = output * 255
    output = np.clip(output, 0, 255).astype(np.uint8)

    cv2.imwrite('input.jpg', origin_img)
    cv2.imwrite('output.jpg', output)


if __name__ == '__main__':
    demo()
