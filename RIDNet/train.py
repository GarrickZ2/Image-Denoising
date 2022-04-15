from cmath import inf
import os
import random
import argparse
from tqdm import tqdm

import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchsummary import summary

from model.RIDNet import RIDNet
from datasets.clean_noisy_dataset import *
from util.dataset import *
from util.loss import *
from util.augment import *


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


def train():
    parser = argparse.ArgumentParser(description='argparse argument')
    parser.add_argument('--epochs',
                        type=int,
                        help='epoch',
                        default='300',
                        dest='epochs')

    parser.add_argument('--batch_size',
                        type=int,
                        help='batch_size',
                        default='8',
                        dest='batch_size')

    args = parser.parse_args()

    # hyper parameters
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_transform = transforms.Compose([
        Resize(),
        Random_Brightness(p=0.5,
                          sigma1=0.3),
        Horizontal_Flip(p=0.5),
        Vertical_Flip(p=0.5),
        Shift_X(p=0.5,
                dx=30),
        Shift_Y(p=0.5,
                dy=30),
        Rotation(p=0.5,
                 angle=(-30, 30)),
        Random_Crop(patch_size=(64, 64)),  # for patch-wise training
        Normalize(),
        ToTensor()
    ])

    train_dataset = SIDDDataset(root_dir='../data/SIDD_Small_sRGB_Only', transform=train_transform)
    # train_dataset = Denoising_dataset(img_dir='your dataset path',
    #                                   train_val='train',
    #                                   transform=train_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=0)

    val_transform = transforms.Compose([
        Resize(),
        Normalize(),
        ToTensor()
    ])

    val_dataset = SIDDDataset(root_dir='../data/SIDD_Small_sRGB_Only', transform=val_transform, data_type='val')
    # val_dataset = Denoising_dataset(img_dir='your dataset path',
    #                                 train_val='val',
    #                                 transform=val_transform)

    val_loader = DataLoader(val_dataset,
                            batch_size=int(BATCH_SIZE/2),
                            shuffle=False,
                            num_workers=0)
    print('Loaded Dataset')

    model = RIDNet(in_channels=1, out_channels=1, num_feautres=128)
    model.to(device)
    # summary(model, (1, 512, 512), batch_size=BATCH_SIZE)

    criterion = L1_Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5, verbose=1)

    # tensorboard
    writer = SummaryWriter('runs/')

    best_val_loss = inf
    print('Prepare to train')

    for epoch in range(1, EPOCHS + 1):
        train_loss = 0.
        val_loss = 0.

        loop = tqdm(enumerate(train_loader), total=len(train_loader))

        model.train()
        for i, data in loop:
            noisy = data['noisy'].to(device)
            clean = data['clean'].to(device)

            optimizer.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, clean)  # pred, gt
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_description(f'Epoch [{epoch}/{EPOCHS}, train loss: {loss.item()}')

        current_lr = scheduler.optimizer.param_groups[0]['lr']
        writer.add_scalar('lr', current_lr, epoch)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            loop = tqdm(enumerate(val_loader), total=len(val_loader))

            for j, data in loop:
                noisy = data['noisy'].to(device)
                clean = data['clean'].to(device)

                pred = model(noisy)

                loss = criterion(pred, clean)
                val_loss += loss.item()
                loop.set_description(f'Epoch [{epoch}/{EPOCHS} validation error: {loss.item()}')

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        if best_val_loss > val_loss:
            # print('=' * 100)
            print('=' * 100)
            print(f'val_loss is improved from {best_val_loss:.8f} to {val_loss:.8f}\t saved current weight')
            print('=' * 100)
            best_val_loss = val_loss

            # torch.save(model, 'model.pth')
            if not os.path.exists('weight'):
                os.mkdir('weight')

            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion},
                       f'weight/{str(criterion).split("()")[0]}_model_{epoch:05d}_valloss_{best_val_loss:.4f}.pth')
        print(f'Epoch: {epoch}\t train_loss: {train_loss}\t val_loss: {val_loss} best_val_loss: {best_val_loss}')

    writer.close()


if __name__ == '__main__':
    seed_everything(42)
    train()
