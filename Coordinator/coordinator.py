import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from PNGAN.model.generator import *
from PNGAN.model.discriminator import *
import torch
from threading import Condition
import os
from asyncio import Lock
from datetime import datetime


class Coordinator:
    def __init__(self, machines=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.netD = Discriminator().to(self.device)
        self.netG = Generator().to(self.device)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=2e-4, betas=(0.9, 0.9999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=2e-4, betas=(0.9, 0.9999))
        self.schedulerD = lr_scheduler.CosineAnnealingLR(self.optimizerD, T_max=7e5, eta_min=1e-6)
        self.schedulerG = lr_scheduler.CosineAnnealingLR(self.optimizerG, T_max=7e5, eta_min=1e-6)

        self.machine_num = machines
        self.update_conditionG = Condition()
        self.update_conditionD = Condition()
        self.update_conditionV = Condition()

        self.g_machine = 0
        self.d_machine = 0
        self.val_machine = 0

        self.ts = int(datetime.timestamp(datetime.now()))

        self.val_loss = 0
        self.val_num = 0
        self.lock = Lock()

        self.history = {
            'train_lossG': [],
            'train_lossD': [],
            'val_loss': [],
            'step_G': 0,
            'step_D': 0
        }

    def update_generator(self, data: torch.Tensor):
        self.update_conditionG.acquire()
        self.g_machine += 1
        if self.g_machine == 1:
            self.optimizerG.zero_grad()
            loss = data / self.machine_num
            self.history['train_lossG'].append(loss.item())
            loss.backward()
            self.update_conditionG.wait()
        elif self.g_machine < self.machine_num:
            loss = data / self.machine_num
            self.history['train_lossG'][-1] += loss.item()
            loss.backward()
            self.update_conditionG.wait()
        else:
            loss = data / self.machine_num
            self.history['train_lossG'][-1] += loss.item()
            loss.backward()
            self.history['step_G'] += 1
            self.optimizerG.step()
            self.update_conditionG.notifyAll()
        self.g_machine -= 1
        state = self.netG.state_dict()
        self.update_conditionG.release()
        return state

    def update_discriminator(self, data: torch.Tensor):
        self.update_conditionD.acquire()
        self.d_machine += 1
        if self.d_machine == 1:
            loss = data / self.machine_num
            self.optimizerD.zero_grad()
            self.history['train_lossD'].append(loss.item())
            loss.backward()
            self.update_conditionD.wait()
        elif self.d_machine < self.machine_num:
            loss = data / self.machine_num
            self.history['train_lossD'][-1] += loss.item()
            loss.backward()
            self.update_conditionD.wait()
        else:
            loss = data / self.machine_num
            self.history['train_lossD'][-1] += loss.item()
            loss.backward()
            self.history['step_D'] += 1
            self.optimizerD.step()
            self.update_conditionD.notifyAll()
        self.d_machine -= 1
        state = self.netD.state_dict()
        self.update_conditionD.release()
        return state

    def update_val(self, data):
        await self.lock.acquire()
        self.val_loss += data
        self.val_num += 1
        self.lock.release()

    def finish_val(self):
        self.update_conditionV.acquire()
        self.val_machine += 1
        if self.val_machine < self.machine_num:
            self.update_conditionV.wait()
        else:
            self.update_conditionV.notifyAll()
        self.val_machine -= 1
        loss = self.val_loss / self.val_num
        if self.val_machine == 0:
            self.val_num = 0
            self.val_loss = 0
        self.update_conditionV.release()
        return loss

    def call_scheduler(self):
        self.schedulerG.step()
        self.schedulerD.step()

    def __load_state(self, store_dic):
        self.history = store_dic['history']
        self.netD.load_state_dict(store_dic['modelD'])
        self.netG.load_state_dict(store_dic['modelG'])
        self.schedulerD.load_state_dict(store_dic['schedD'])
        self.schedulerG.load_state_dict(store_dic['schedG'])
        self.optimizerG.load_state_dict(store_dic['optimG'])
        self.optimizerD.load_state_dict(store_dic['optimD'])

    def load(self, dir_path, ts):
        model_path = f"{dir_path}/result/{ts}"
        checkpoint = torch.load(f'{model_path}/model.pt')
        self.__load_state(checkpoint)
        print('Load Model from ts: ', ts)

    def save(self, dir_path):
        if not os.path.exists(f'{dir_path}/result'):
            os.mkdir(f'{dir_path}/result')
        if not os.path.exists(f'{dir_path}/result/{self.ts}'):
            os.mkdir(f'{dir_path}/result/{self.ts}')
        model_path = f'{dir_path}/result/{self.ts}/model.pt'
        store_dictory = {
            'modelD': self.netD.state_dict(),
            'modelG': self.netG.state_dict(),
            'schedG': self.schedulerG.state_dict(),
            'schedD': self.schedulerD.state_dict(),
            'optimG': self.optimizerG.state_dict(),
            'optimD': self.optimizerD.state_dict(),
            'history': self.history
        }
        torch.save(store_dictory, model_path)
        print('Saved Model at ts: ', self.ts)
