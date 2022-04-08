from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
import torch
import numpy as np
import unittest
import utility
from option import args
from model import RIDModel
from model.generator import *
from model.discriminator import Discriminator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestFlow(unittest.TestCase):
    def test_discriminator(self):
        print('Test Discriminator')
        input_data = torch.rand(1, 3, 128, 128, device=device)
        model_d = Discriminator()
        output_data = model_d(input_data)
        print('Input Data: ', input_data, input_data.shape)
        print('Output Data: ', output_data, output_data.shape)

    def test_BlurPool2D(self):
        print('Test BLUR POOL 2D')
        input_data = torch.rand(1, 3, 256, 256, device=device)
        blur_pool_2d = BlurPool2d(channels=3, filt_size=3, stride=4)
        output_data = blur_pool_2d(input_data)
        print('Input Data Size: ', input_data.shape)
        print('Output Data Size: ', output_data.shape)

    def test_up_sample(self):
        # k_size = 3 for scale 2; k_size = 5 for scale 4
        print('Test FAC')
        input_data = torch.rand(1, 64, 64, 64, device=device)
        up_sample_model = up_sample(64, scale=2, k_size=3)
        output_data = up_sample_model(input_data)
        print('Input Data Size: ', input_data.shape)
        print('Output Data Size: ', output_data.shape)

    def test_fac(self):
        print('Test FAC')
        input_data = torch.rand(1, 64, 128, 128, device=device)
        fac = FCA()
        output_data = fac(input_data)
        print('Input Data Size: ', input_data.shape)
        print('Output Data Size: ', output_data.shape)

    def test_mab(self):
        print('Test MAB')
        input_data = torch.rand(1, 64, 128, 128, device=device)
        mab = MAB(64)
        output_data = mab(input_data)
        print('Input Data Size: ', input_data.shape)
        print('Output Data Size: ', output_data.shape)

    def test_srb(self):
        print('Test SRB')
        input_data = torch.rand(1, 64, 128, 128, device=device)
        srg = SRG(64, 64)
        output_data = srg(input_data)
        print('Input Data Size: ', input_data.shape)
        print('Output Data Size: ', output_data.shape)

    def test_generator(self):
        input_data = torch.rand(1, 3, 128, 128, device=device)
        generator = Generator(3, 64)
        output_data = generator(input_data)
        print('Input Data Size: ', input_data.shape)
        print('Output Data Size: ', output_data.shape)


class TestRIDNet(unittest.TestCase):
    def test_ridmodel(self):
        img = Image.open('a.png')
        img = img.convert('RGB')
        img = np.array(img.getdata()).reshape((img.size[0], img.size[1], 3))
        plt.imshow(img)
        plt.show()
        # input_data = torch.rand(1, 3, 128, 128, device=device)
        input_data = torch.from_numpy(img)
        # checkpoint = utility.checkpoint(args)
        # model = RIDModel(args, checkpoint)
        # output_data = model(input_data, 10)
        # print('Input Data Size: ', input_data.shape)
        # print('Output Data Size: ', output_data.shape)
        # print(output_data)
        # plt.imshow(output_data)
        # plt.show()


