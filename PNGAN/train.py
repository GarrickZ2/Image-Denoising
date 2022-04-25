import os
from datetime import datetime

import torch
import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import json


class Trainer:
    def __init__(self, netG, netD, train_set, val_set, criterion, optimG, optimD, schedG, schedD, device):
        self.history = {
            'train_loss_G': [],
            'train_loss_D': [],
            'val_loss': [],
            'epoch': 0
        }
        self.criterion = criterion
        self.trainset = train_set
        self.valset = val_set
        self.train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=8)
        self.device = device
        self.optimG = optimG
        self.optimD = optimD
        self.netD = netD
        self.netG = netG
        self.schedG = schedG
        self.schedD = schedD
        self.ts = int(datetime.timestamp(datetime.now()))

    def train_generator_step(self, irns, isyns):
        self.netD.eval()
        self.netG.train(mode=True)
        self.optimG.zero_grad()
        ifns = self.netG(isyns)
        _, cd_irns = self.netD(irns)
        _, cd_ifns = self.netD(ifns)
        lossG = self.criterion(irns, ifns, cd_irns, cd_ifns)
        lossG.backward()
        self.optimG.step()
        return lossG.item()

    def train_discriminator_step(self, irns, isyns):
        self.netG.eval()
        self.netD.train(mode=True)
        self.optimD.zero_grad()
        ifns = self.netG(isyns)
        _, cd_irns = self.netD(irns)
        _, cd_ifns = self.netD(ifns)
        lossD = self.criterion(irns, ifns, cd_irns, cd_ifns)
        lossD.backward()
        self.optimD.step()
        return lossD.item()

    def val_step(self, irns, isyns, netG, netD):
        ifns = netG(isyns)
        _, cd_irns = netD(irns)
        _, cd_ifns = netD(ifns)
        loss = self.criterion(irns, ifns, cd_irns, cd_ifns)
        return loss.item()

    def plot(self, dir_path, show=False):
        if not os.path.exists(f'{dir_path}/result'):
            os.mkdir(f'{dir_path}/result')
        if not os.path.exists(f'{dir_path}/result/{self.ts}'):
            os.mkdir(f'{dir_path}/result/{self.ts}')

        plt.plot(self.history['train_loss_G'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train Loss for Generator')
        if show:
            plt.show()
        plt.savefig(f'{dir_path}/result/{self.ts}/loss_g.jpg')
        plt.close()

        plt.plot(self.history['train_loss_D'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train Loss for Discriminator')
        if show:
            plt.show()
        plt.savefig(f'{dir_path}/result/{self.ts}/loss_d.jpg')
        plt.close()

        plt.plot(self.history['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train Loss for Validation')
        if show:
            plt.show()
        plt.savefig(f'{dir_path}/result/{self.ts}/loss_val.jpg')
        plt.close()

        print(f'Figure Saved in directory {dir_path}/result/{self.ts}')

    def save(self, dir_path):
        if not os.path.exists(f'{dir_path}/result'):
            os.mkdir(f'{dir_path}/result')
        if not os.path.exists(f'{dir_path}/result/{self.ts}'):
            os.mkdir(f'{dir_path}/result/{self.ts}')
        torch.save(self.netD.state_dict(), f'{dir_path}/result/{self.ts}/modelD_{self.history["epoch"]}.pt')
        torch.save(self.netG.state_dict(), f'{dir_path}/result/{self.ts}/modelG_{self.history["epoch"]}.pt')
        with open(f'{dir_path}/result/{self.ts}/history_{self.history["epoch"]}.json', 'w') as f:
            json.dump(self.history, f)

        torch.save(self.netD.state_dict(), f'{dir_path}/result/{self.ts}/modelD.pt')
        torch.save(self.netG.state_dict(), f'{dir_path}/result/{self.ts}/modelG.pt')
        with open(f'{dir_path}/result/{self.ts}/history.json', 'w') as f:
            json.dump(self.history, f)

    def load_latest(self, dir_path, ts=None):
        model_path = dir_path
        if ts is not None:
            model_path = f"{dir_path}/result/{ts}"
        self.netD.load_state_dict(torch.load(f'{model_path}/modelD.pt'))
        self.netG.load_state_dict(torch.load(f'{model_path}/modelG.pt'))
        with open(f'{model_path}/history.json', 'r') as f:
            self.history = json.load(f)
        print('Load Model from epoch: ', self.history['epoch'])

    def load(self, dir_path, ts=None, epoch_num=None):
        if epoch_num is None:
            self.load_latest(dir_path, ts)
            return
        model_path = dir_path
        if ts is not None:
            model_path = f"{dir_path}/result/{ts}"
        self.netD.load_state_dict(torch.load(f'{model_path}/modelD_{epoch_num}.pt'))
        self.netG.load_state_dict(torch.load(f'{model_path}/modelG_{epoch_num}.pt'))
        with open(f'{model_path}/history_{epoch_num}.json', 'r') as f:
            self.history = json.load(f)
        print('Load Model from epoch: ', epoch_num)

    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            train_loss_G = 0.0
            train_loss_D = 0.0
            process = tqdm.tqdm(self.train_loader, leave=False)
            for i, (_, irns, isyns) in enumerate(process):
                irns = irns.to(self.device)
                isyns = isyns.to(self.device)
                step_loss_G = self.train_generator_step(irns, isyns)
                step_loss_D = self.train_discriminator_step(irns, isyns)
                train_loss_G += step_loss_G
                train_loss_D += step_loss_D
                process.set_description(
                    f"Epoch {epoch + 1}: generator_train_loss={step_loss_G}, discriminator_train_loss={step_loss_D}")

            train_loss_G /= len(self.trainset)
            train_loss_D /= len(self.trainset)
            self.history['train_loss_G'].append(train_loss_G)
            self.history['train_loss_D'].append(train_loss_D)
            process.close()

            self.netD.eval()
            self.netG.eval()
            val_loss = 0.0
            process = tqdm.tqdm(self.val_loader, leave=False)
            for i, (_, irns, isyns) in enumerate(process):
                irns = irns.to(self.device)
                isyns = isyns.to(self.device)
                val_loss += self.val_step(irns, isyns, self.netG, self.netD)

            val_loss /= len(self.valset)
            self.history['val_loss'].append(val_loss)
            self.history['epoch'] += 1
            print(
                f"Epoch {self.history['epoch'] + 1}: val_loss={val_loss} generator_train_loss={train_loss_G}, discriminator_train_loss={train_loss_D}")

            self.schedD.step()
            self.schedG.step()
