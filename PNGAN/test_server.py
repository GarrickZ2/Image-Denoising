import requests
import pickle
import base64
from model.generator import *
from model.discriminator import *
from loss.loss import AlignmentLoss
import json

netG = Generator()
netD = Discriminator()
criterion = AlignmentLoss()

netG.train(mode=True)
netD.eval()
isyns = torch.rand(1, 3, 128, 128)
irns = torch.rand(1, 3, 128, 128)
ifns = netG(isyns)
_, cd_irns = netD(irns)
_, cd_ifns = netD(ifns)
lossG = criterion(irns, ifns, cd_irns, cd_ifns)

data = pickle.dumps(lossG)
data = base64.b64encode(data)

send_data = {'loss': data}
url = 'http://127.0.0.1:6000/generator'
response = requests.post(url, data=send_data).content
state_dict = pickle.loads(response)
netG.load_state_dict(state_dict)
print('Success')
