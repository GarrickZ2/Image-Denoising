import matplotlib

matplotlib.use('TkAgg')
import unittest
from model.generator import *
from model.discriminator import Discriminator
from loss.loss import AlignmentLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestFlow(unittest.TestCase):
    def test_discriminator(self):
        print('Test Discriminator')
        input_data = torch.rand(1, 3, 128, 128, device=device)
        model_d = Discriminator()
        output_data, _ = model_d(input_data)
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
    def test_loss(self):
        input_data_1 = torch.rand(1, 3, 128, 128, device=device)
        input_data_2 = torch.rand(1, 3, 128, 128, device=device)
        model_d = Discriminator()
        output_1, _ = model_d(input_data_1)
        loss = AlignmentLoss(model_d)
        output_data = loss(input_data_1, input_data_2)
        output_data.backward()
        print(output_data.shape)
        print(output_data)
