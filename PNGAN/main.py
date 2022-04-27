from util.option import args
from model.generator import *
from model.discriminator import *
from loss.loss import AlignmentLoss
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from PNGAN.dataset.dataset import *
from PNGAN.train import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args.pre_train = './experiment/ridnet.pt'
train_ds = SIDDSmallDataset('./Datasets', noise_generator=AdditiveGaussianWhiteNoise())
val_ds = SIDDSmallDataset('./Datasets', data_type='val', noise_generator=AdditiveGaussianWhiteNoise())
print(len(train_ds), len(val_ds))

netD = Discriminator().to(device)
netG = Generator(3, 64).to(device)
criterion = AlignmentLoss().to(device)

optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.9, 0.9999))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.9, 0.9999))

schedulerD = lr_scheduler.CosineAnnealingLR(optimizerD, T_max=7e5, eta_min=1e-6)
schedulerG = lr_scheduler.CosineAnnealingLR(optimizerG, T_max=7e5, eta_min=1e-6)

train_process = Trainer(netG, netD, train_ds, val_ds, criterion, optimizerG, optimizerD, schedulerG, schedulerD, device)

train_process.train('./', num_epochs=1)
train_process.save('./')
