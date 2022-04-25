import cv2
import numpy as np
import glob
import pandas as pd
import os
from tqdm import tqdm

src_input = 'Datasets/val/SIDD/input_crops'
src_target = 'Datasets/val/SIDD/target_crops'

input_files = pd.Series(glob.glob(f"{src_input}/**"))
target_files = pd.Series(glob.glob(f"{src_target}/**"))

patch_size = 128
overlap = 32
p_max = 0


def save_files(lr_file, hr_file):
    filename = os.path.splitext(os.path.split(lr_file)[-1])[0]
    lr_img = cv2.imread(lr_file)
    hr_img = cv2.imread(hr_file)
    num_patch = 0
    w, h = lr_img.shape[:2]
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w - patch_size, patch_size - overlap, dtype=np.int32))
        h1 = list(np.arange(0, h - patch_size, patch_size - overlap, dtype=np.int32))
        w1.append(w - patch_size)
        h1.append(h - patch_size)
        for i in w1:
            for j in h1:
                num_patch += 1

                lr_patch = lr_img[i:i + patch_size, j:j + patch_size, :]
                hr_patch = hr_img[i:i + patch_size, j:j + patch_size, :]

                lr_savename = os.path.join(src_input, filename + '-' + str(num_patch) + '.png')
                hr_savename = os.path.join(src_target, filename + '-' + str(num_patch) + '.png')

                cv2.imwrite(lr_savename, lr_patch)
                cv2.imwrite(hr_savename, hr_patch)

    else:
        lr_savename = os.path.join(src_input, filename + '.png')
        hr_savename = os.path.join(src_target, filename + '.png')

        cv2.imwrite(lr_savename, lr_img)
        cv2.imwrite(hr_savename, hr_img)

    os.remove(lr_file)
    os.remove(hr_file)


from joblib import Parallel, delayed
import multiprocessing

num_cores = 10
Parallel(n_jobs=num_cores)(delayed(save_files)(input_files[i], target_files[i]) for i in tqdm(range(len(input_files))))
