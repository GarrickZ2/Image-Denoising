import os
from datetime import datetime
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


class Trainer:
    def __init__(self, netG, netD, train_set, val_set, criterion_d, criterion_g, performance, optimD, optimG, schedD, schedG, device, batch=8):
        self.history = {
            'train_loss_G': [],
            'train_loss_D': [],
            'train_loss_G_epoch': [],
            'train_loss_D_epoch': [],
            'performance': [],
            'val_loss': [],
            'epoch': 0,
            'best_val_loss': np.inf,
            'step': 0,
        }
        self.batch = batch
        self.criterion_d = criterion_d
        self.criterion_g = criterion_g
        self.performance = performance
        self.trainset = train_set
        self.valset = val_set
        self.train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_set, batch_size=batch, num_workers=4)
        self.device = device
        self.optimG = optimG
        self.optimD = optimD
        self.netD = netD
        self.netG = netG
        self.schedG = schedG
        self.schedD = schedD
        self.ts = int(datetime.timestamp(datetime.now()))

    def __train_generator_step(self, irns, isyns):
        self.optimG.zero_grad()
        ifns = self.netG(isyns)
        _, cd_irns = self.netD(irns)
        _, cd_ifns = self.netD(ifns)
        loss_g = self.criterion_g(irns, ifns, cd_irns, cd_ifns)
        loss_g.backward()
        self.optimG.step()
        return loss_g.item()

    def __train_discriminator_step(self, irns, isyns):
        self.optimD.zero_grad()
        ifns = self.netG(isyns)
        _, cd_irns = self.netD(irns)
        _, cd_ifns = self.netD(ifns)
        loss_d = self.criterion_d(cd_irns, cd_ifns)
        loss_d.backward()
        self.optimD.step()
        return loss_d.item(), ifns

    def __val_step(self, irns, isyns, netG, netD):
        ifns = netG(isyns)
        _, cd_irns = netD(irns)
        _, cd_ifns = netD(ifns)
        loss_d = self.criterion_d(cd_irns, cd_ifns)
        loss_g = self.criterion_g(irns, ifns, cd_irns, cd_ifns)
        performance = self.performance(irns, ifns, loss_d, loss_g)
        return loss_d.item(), loss_g.item(), performance.item()

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

    def save(self, dir_path, best=False):
        if not os.path.exists(f'{dir_path}/result'):
            os.mkdir(f'{dir_path}/result')
        if not os.path.exists(f'{dir_path}/result/{self.ts}'):
            os.mkdir(f'{dir_path}/result/{self.ts}')
        model_path = f'{dir_path}/result/{self.ts}/epoch_{self.history["epoch"]}_{self.history["step"]}.pt'
        store_dictory = {
            'modelD': self.netD.state_dict(),
            'modelG': self.netG.state_dict(),
            'schedG': self.schedG.state_dict(),
            'schedD': self.schedD.state_dict(),
            'optimG': self.optimG.state_dict(),
            'optimD': self.optimD.state_dict(),
            'history': self.history
        }
        torch.save(store_dictory, model_path)
        torch.save(store_dictory, f'{dir_path}/result/{self.ts}/model.pt')
        if best:
            torch.save(store_dictory, f'{dir_path}/result/{self.ts}/model_best.pt')

    def __load_state(self, store_dic):
        self.history = store_dic['history']
        self.netD.load_state_dict(store_dic['modelD'])
        self.netG.load_state_dict(store_dic['modelG'])
        self.schedD.load_state_dict(store_dic['schedD'])
        self.schedG.load_state_dict(store_dic['schedG'])
        self.optimG.load_state_dict(store_dic['optimG'])
        self.optimD.load_state_dict(store_dic['optimD'])

    def load_latest(self, dir_path, ts=None, best=False):
        model_path = dir_path
        if ts is not None:
            model_path = f"{dir_path}/result/{ts}"
        if best:
            print('Loading The Best Model')
            checkpoint = torch.load(f'{model_path}/model_best.pt')
        else:
            checkpoint = torch.load(f'{model_path}/model.pt')
        self.__load_state(checkpoint)
        print('Load Model from epoch: ', self.history['epoch'])

    def load(self, dir_path, ts=None, epoch_num=None):
        if epoch_num is None:
            self.load_latest(dir_path, ts)
            return
        model_path = dir_path
        if ts is not None:
            model_path = f"{dir_path}/result/{ts}"
        checkpoint = torch.load(f'{model_path}/epoch_{epoch_num}.pt')
        self.__load_state(checkpoint)
        print('Load Model from epoch: ', epoch_num)

    def train(self, dir_path, num_epochs=10):
        for epoch in range(num_epochs):
            self.netD.train(mode=True)
            self.netG.train(mode=True)
            train_loss_G = 0.0
            train_loss_D = 0.0
            process = tqdm.tqdm(self.train_loader)
            for i, (_, irns, isyns) in enumerate(process):
                if i < self.history['step']:
                    continue
                self.history['step'] = i
                irns = irns.to(self.device)
                isyns = isyns.to(self.device)
                step_loss_G = self.__train_generator_step(irns, isyns)
                step_loss_D, ifns = self.__train_discriminator_step(irns, isyns)
                train_loss_G += step_loss_G
                train_loss_D += step_loss_D
                performance = self.performance(irns, ifns, step_loss_G, step_loss_D)
                self.history['train_loss_G'] = step_loss_G / self.batch
                self.history['train_loss_D'] = step_loss_D / self.batch
                self.history['performance'] = performance / self.batch

                process.set_description(
                    f"Epoch {epoch + 1}:performance:{performance/self.batch},"
                    f" generator_train_loss={step_loss_G/self.batch}, discriminator_train_loss={step_loss_D/self.batch}")
                self.schedD.step()
                self.schedG.step()
                if self.history['step'] % 2000 == 0:
                    print('Saved the model for step', self.history['step'])
                    self.save(dir_path)

            train_loss_G /= len(self.trainset)
            train_loss_D /= len(self.trainset)
            self.history['train_loss_G_epoch'].append(train_loss_G)
            self.history['train_loss_D_epoch'].append(train_loss_D)
            process.close()

            self.netD.eval()
            self.netG.eval()
            val_ld_loss = 0.0
            val_lg_loss = 0.0
            val_performance = 0.0
            process = tqdm.tqdm(self.val_loader)
            for i, (_, irns, isyns) in enumerate(process):
                irns = irns.to(self.device)
                isyns = isyns.to(self.device)
                ld_loss, lg_loss, performance = self.__val_step(irns, isyns, self.netG, self.netD)
                val_ld_loss += ld_loss
                val_lg_loss += lg_loss
                val_performance += performance

            val_lg_loss /= len(self.valset)
            val_ld_loss /= len(self.valset)
            val_performance /= len(self.valset)
            self.history['val_loss'].append(val_performance)
            self.history['epoch'] += 1
            print(f"Epoch {self.history['epoch'] + 1}: "
                  f"val_d_loss={val_ld_loss}, val_g_loss={val_lg_loss}, val_perf={val_performance}")
            if self.history['epoch'] % 1 == 0:
                self.save(dir_path)
            if val_performance < self.history['best_val_loss']:
                self.save(dir_path, best=True)
                self.history['best_val_loss'] = val_performance

    def generator_predict(self, batch_data):
        self.netG.eval()
        result = self.netG(batch_data)
        self.netG.train(mode=True)
        return result

    def discriminator_predict(self, batch_data):
        self.netD.eval()
        result = self.netD(batch_data)
        self.netD.train(mode=True)
        return result
