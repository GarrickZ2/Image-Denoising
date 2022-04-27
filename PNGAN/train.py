import os
from datetime import datetime
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
import requests
import pickle
import base64
from matplotlib import pyplot as plt


class Trainer:
    def __init__(self, netG, netD, train_set, val_set, criterion, optimG, optimD, schedG, schedD, device, batch=8):
        self.history = {
            'train_loss_G': [],
            'train_loss_D': [],
            'train_loss_G_epoch': [],
            'train_loss_D_epoch': [],
            'val_loss': [],
            'epoch': 0,
            'best_val_loss': np.inf,
            'step': 0,
        }
        self.batch = batch
        self.criterion = criterion
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

    def __train_discriminator_step(self, irns, isyns):
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

    def __val_step(self, irns, isyns, netG, netD):
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

    def finish_val(self, loss):
        if isinstance(self, ParallelTrainer):
            return requests.get(f"{self.url}/validate/finished")
        return loss

    def train(self, dir_path, num_epochs=10):
        self.netG.train(mode=True)
        self.netD.train(mode=True)
        for epoch in range(num_epochs):
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
                step_loss_D = self.__train_discriminator_step(irns, isyns)
                train_loss_G += step_loss_G
                train_loss_D += step_loss_D
                self.history['train_loss_G'] = step_loss_G / self.batch
                self.history['train_loss_D'] = step_loss_D / self.batch
                process.set_description(
                    f"Epoch {epoch + 1}: generator_train_loss={step_loss_G/self.batch}, discriminator_train_loss={step_loss_D/self.batch}")
                self.schedD.step()
                self.schedG.step()
                if self.history['step'] % 2000 == 0:
                    print('Saved the model for step', self.history['step'])
                    self.save(dir_path)
                if i % 200 == 0:
                    torch.cuda.empty_cache()

            train_loss_G /= len(self.trainset)
            train_loss_D /= len(self.trainset)
            self.history['train_loss_G_epoch'].append(train_loss_G)
            self.history['train_loss_D_epoch'].append(train_loss_D)
            process.close()

            self.netD.eval()
            self.netG.eval()
            val_loss = 0.0
            process = tqdm.tqdm(self.val_loader)
            for i, (_, irns, isyns) in enumerate(process):
                irns = irns.to(self.device)
                isyns = isyns.to(self.device)
                val_loss += self.__val_step(irns, isyns, self.netG, self.netD)
                if i % 200 == 0:
                    torch.cuda.empty_cache()

            val_loss = self.finish_val(val_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epoch'] += 1
            print(
                f"Epoch {self.history['epoch'] + 1}: val_loss={val_loss} generator_train_loss={train_loss_G}, "
                f"discriminator_train_loss={train_loss_D}")
            if self.history['epoch'] % 5 == 0:
                self.save(dir_path)
            if val_loss < self.history['best_val_loss']:
                self.save(dir_path, best=True)
                self.history['best_val_loss'] = val_loss

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


class ParallelTrainer(Trainer):
    def __init__(self, netG, netD, train_set, val_set, criterion, optimG, optimD, schedG, schedD, device, url, batch=8):
        super().__init__(netG, netD, train_set, val_set, criterion, optimG, optimD, schedG, schedD, device, batch)
        self.url = url

    def __train_generator_step_parallel(self, irns, isyns):
        self.netD.eval()
        self.netG.train(mode=True)
        self.optimG.zero_grad()
        ifns = self.netG(isyns)
        _, cd_irns = self.netD(irns)
        _, cd_ifns = self.netD(ifns)
        lossG = self.criterion(irns, ifns, cd_irns, cd_ifns)
        data = pickle.dumps(lossG)
        data = base64.b64encode(data)
        send_data = {'loss': data}
        response = requests.post(f"{self.url}/generator", data=send_data).content
        state_dict = pickle.loads(response)
        self.netG.load_state_dict(state_dict)

        return lossG.item()

    def __train_discriminator_step_parallel(self, irns, isyns):
        self.netG.eval()
        self.netD.train(mode=True)
        self.optimD.zero_grad()
        ifns = self.netG(isyns)
        _, cd_irns = self.netD(irns)
        _, cd_ifns = self.netD(ifns)
        lossD = self.criterion(irns, ifns, cd_irns, cd_ifns)

        data = pickle.dumps(lossD)
        data = base64.b64encode(data)
        send_data = {'loss': data}
        response = requests.post(f"{self.url}/discriminator", data=send_data).content
        state_dict = pickle.loads(response)
        self.netD.load_state_dict(state_dict)

        return lossD.item()

    def __val_step_parallel(self, irns, isyns, netG, netD):
        ifns = netG(isyns)
        _, cd_irns = netD(irns)
        _, cd_ifns = netD(ifns)
        loss = self.criterion(irns, ifns, cd_irns, cd_ifns)
        requests.get(f"{self.url}/validate?val={loss.item()}")
        return loss.item()

    def train(self, dir_path, num_epochs=10):
        self.netG.train(mode=True)
        self.netD.train(mode=True)
        for epoch in range(num_epochs):
            train_loss_G = 0.0
            train_loss_D = 0.0
            process = tqdm.tqdm(self.train_loader)
            for i, (_, irns, isyns) in enumerate(process):
                irns = irns.to(self.device)
                isyns = isyns.to(self.device)
                step_loss_G = self.__train_generator_step_parallel(irns, isyns)
                step_loss_D = self.__train_discriminator_step_parallel(irns, isyns)
                train_loss_G += step_loss_G
                train_loss_D += step_loss_D
                process.set_description(
                    f"Epoch {epoch + 1}: generator_train_loss={step_loss_G}, discriminator_train_loss={step_loss_D}")
                self.schedD.step()
                self.schedG.step()
                if i % 200 == 0:
                    torch.cuda.empty_cache()

            train_loss_G /= len(self.trainset)
            train_loss_D /= len(self.trainset)
            self.history['train_loss_G'].append(train_loss_G)
            self.history['train_loss_D'].append(train_loss_D)
            process.close()

            self.netD.eval()
            self.netG.eval()
            val_loss = 0.0
            process = tqdm.tqdm(self.val_loader)
            for i, (_, irns, isyns) in enumerate(process):
                irns = irns.to(self.device)
                isyns = isyns.to(self.device)
                val_loss += self.__val_step_parallel(irns, isyns, self.netG, self.netD)
                if i % 200 == 0:
                    torch.cuda.empty_cache()

            val_loss = self.finish_val(val_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epoch'] += 1
            print(
                f"Epoch {self.history['epoch'] + 1}: val_loss={val_loss} generator_train_loss={train_loss_G}, "
                f"discriminator_train_loss={train_loss_D}")

            if self.history['epoch'] % 5 == 0:
                self.save(dir_path)
            if val_loss < self.history['best_val_loss']:
                self.save(dir_path, best=True)
                self.history['best_val_loss'] = val_loss