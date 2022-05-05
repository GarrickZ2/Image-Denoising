import torch
from loss.loss import *
from train import Trainer
import torch.optim as optim
from util.option import args
from dataset.dataset import *
from model.generator import *
from model.discriminator import *
import torch.optim.lr_scheduler as lr_scheduler

print('Configuration:')
print(args)
print('=================')

torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.partial_data:
    train_ds = SIDDSmallDataset(args.dir_data, noise_generator=AdditiveGaussianWhiteNoise(std=args.noise),
                                random_load=True, limit=args.n_train)
    val_ds = SIDDSmallDataset(args.dir_data, data_type='val', noise_generator=AdditiveGaussianWhiteNoise(std=args.noise),
                              random_load=True, limit=args.n_val)
else:
    train_ds = SIDDSmallDataset(args.dir_data, noise_generator=AdditiveGaussianWhiteNoise(std=args.noise))
    val_ds = SIDDSmallDataset(args.dir_data, data_type='val', noise_generator=AdditiveGaussianWhiteNoise(std=args.noise))


netD = Discriminator().to(device)
netG = Generator(args.n_colors, args.n_feats).to(device)

criterion_d = DLoss().to(device)
criterion_g = GLoss().to(device)

optimizer_d = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_g = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

scheduler_d = lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=int(args.lr_decay_step), eta_min=args.lr_min)
scheduler_g = lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=int(args.lr_decay_step), eta_min=args.lr_min)

train_process = Trainer(netD, netG, train_ds, val_ds, criterion_d, criterion_g, optimizer_d, optimizer_g,
                        scheduler_d, scheduler_g, device, batch=args.batch_size)

if args.load_models:
    if args.load_best:
        train_process.load_latest(args.load_dir, args.timestamp, True)
    else:
        train_process.load(args.load_dir, args.timestamp, args.load_epoch)

if args.test_only:
    train_process.predict_dir(args.testpath, args.predict_patch_size, args.savepath)
    exit(0)

train_process.train(args.save, num_epochs=args.epochs)

if args.save_models:
    train_process.save(args.save)
