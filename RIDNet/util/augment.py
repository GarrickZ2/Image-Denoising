import copy
import cv2
import numpy as np
import torch


class Resize(object):
    def __init__(self, size=(512, 512)):
        self.size = size

    def __call__(self, data):
        noisy, clean = data['noisy'], data['clean']

        noisy = cv2.resize(noisy, self.size, interpolation=cv2.INTER_AREA)
        clean = cv2.resize(clean, self.size, interpolation=cv2.INTER_AREA)

        return {'noisy': noisy, 'clean': clean}


class ToTensor(object):
    def __call__(self, data):        
        noisy, clean = data['noisy'], data['clean']

        # (512, 512) -> (512, 512, 1)
        noisy = np.expand_dims(noisy, -1)
        clean = np.expand_dims(clean, -1)
        
        noisy = torch.from_numpy(noisy.copy()).type(torch.float32)
        clean = torch.from_numpy(clean.copy()).type(torch.float32)
                
        # (H, W, C) -> (C, H, W)
        noisy = noisy.permute(2, 0, 1)
        clean = clean.permute(2, 0, 1)        

        data = {'noisy': noisy, 'clean': clean}

        return data


class Normalize(object):
    def __call__(self, data):
        noisy, clean = data['noisy'], data['clean']
        
        noisy = noisy / 255.
        clean = clean / 255.        

        data = {'noisy': noisy, 'clean': clean}

        return data


class Random_Brightness(object):
    def __init__(self, p, sigma1):
        self.p = p
        self.sigma1 = sigma1

    def __call__(self, data):
        noisy, clean = data['noisy'], data['clean']

        if self.p >= np.random.random():
            self.sigma1 = np.random.uniform(low=-(self.sigma1), high=(self.sigma1)) # e.g.  -0.3 ~ 0.3            
            noisy = cv2.add(noisy, np.mean(noisy)*self.sigma1)            

        data = {'noisy': noisy, 'clean': clean}

        return data


class Horizontal_Flip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        noisy, clean = data['noisy'], data['clean']
       
        if np.random.rand() <= self.p:
            noisy = noisy[:, ::-1]
            clean = clean[:, ::-1]
          
        data = {'noisy': noisy, 'clean': clean}

        return data


class Vertical_Flip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        noisy, clean = data['noisy'], data['clean']
      
        if np.random.rand() <= self.p:
            noisy = noisy[::-1, :]
            clean = clean[::-1, :]            
           
        data = {'noisy': noisy, 'clean': clean}

        return data


class Rotation(object):   
    def __init__(self, p=0.5, angle=(-30, 30)):
        self.p = p 
        self.angle = angle        

    def __call__(self, data):
        noisy, clean = data['noisy'], data['clean']

        if self.p >= np.random.random():
            h, w = clean.shape
            rotation_angle = np.random.randint(self.angle[0], self.angle[1])
            rotation_matrix = cv2.getRotationMatrix2D((h/2, w/2), rotation_angle, 1)
            
            noisy = cv2.warpAffine(noisy, rotation_matrix, (h, w))
            clean = cv2.warpAffine(clean, rotation_matrix, (h, w))            
        
        data = {'noisy': noisy, 'clean': clean}

        return data


class Shift_X(object):
    def __init__(self, p, dx=30):
        self.p = p        
        self.dx = np.random.randint(low=-dx, high=dx)

    def __call__(self, data):
        noisy, clean = data['noisy'], data['clean']

        if self.p >= np.random.random():
            h, w = clean.shape
            shifted_noisy = np.zeros(noisy.shape).astype(np.uint8)
            shifted_clean = np.zeros(clean.shape).astype(np.uint8)            
            
            if self.dx > 0: # shift right
                shifted_noisy[:, self.dx:] = noisy[:, :w-self.dx]
                shifted_clean[:, self.dx:] = clean[:, :w-self.dx]
            else: # shift left                
                shifted_noisy[:, :w+self.dx] = noisy[:, (-self.dx):]
                shifted_clean[:, :w+self.dx] = clean[:, (-self.dx):]

            data = {'noisy': shifted_noisy, 'clean': shifted_clean}            
        else:
            data = {'noisy': noisy, 'clean': clean}

        return data


class Shift_Y(object):
    def __init__(self, p, dy=30):
        self.p = p        
        self.dy = np.random.randint(low=-dy, high=dy)

    def __call__(self, data):
        noisy, clean = data['noisy'], data['clean']

        if self.p >= np.random.random():
            h, w = clean.shape
            shifted_noisy = np.zeros(noisy.shape).astype(np.uint8)
            shifted_clean = np.zeros(clean.shape).astype(np.uint8)            
            
            if self.dy > 0: # shift up
                shifted_noisy[:h-self.dy, :] = noisy[self.dy:, :]
                shifted_clean[:h-self.dy, :] = clean[self.dy:, :]                
            else: # shift down
                shifted_noisy[-self.dy:, :] = noisy[:(h+self.dy), :]
                shifted_clean[-self.dy:, :] = clean[:(h+self.dy), :]                

            data = {'noisy': shifted_noisy, 'clean': shifted_clean}            
        else:
            data = {'noisy': noisy, 'clean': clean}

        return data


class Random_Crop(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size        

    def __call__(self, data):
        noisy, clean = data['noisy'], data['clean']

        h, w = clean.shape

        top = np.random.randint(0, h - self.patch_size[0])
        bottom = top + self.patch_size[0]        
        left = np.random.randint(0, w - self.patch_size[1])
        right = left + self.patch_size[1]
        
        noisy_patch = noisy[top:bottom, left:right]
        clean_patch = clean[top:bottom, left:right]
   
        data = {'noisy': noisy_patch, 'clean': clean_patch}

        return data
