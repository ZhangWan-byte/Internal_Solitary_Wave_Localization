import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from dataset import *
from metric import *
from model import *
from train_eval import *
from loss_func import *

from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

import numpy as np
from tqdm import tqdm

def myTransform(x):
    x = torch.tensor(x).to(torch.float32)
    
    return x


class myTransformsSimCLR:

    def __init__(self, size):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # probability 0.5
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
            ]
        )
        
    def __call__(self, x):
        x = myTransform(x)
        return self.train_transform(x), self.train_transform(x)


# load global test data (unlabeleld data)
global_test = np.load("./data/global_southSea_data.npy")
print(global_test.shape)


epochs = 1#10
batch_size = 128 #256 #512 #1024
lr = 3e-4
temperature = 100 # 0.1
world_size = 1
device = torch.device("cuda")

transform = myTransformsSimCLR(size=16)
train = myDataset(data=global_test, label=None, is_global_test=True, alter_channel=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset = train, batch_size=batch_size)

encoder = ResNet50() # BoTNet50()
projection_dim = 64
n_features = 2048 # get dimensions of last fully-connected layer
model = SimCLR(encoder, projection_dim, n_features).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)
criterion = NT_Xent(batch_size, temperature, world_size)

history_train_loss = []
min_loss = 100
model.train()

for epoch in range(epochs):
    for (x_i, x_j), _ in tqdm(train_loader):
        if x_i.shape[0]!=batch_size:
            break
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True) # non_blocking=True

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()
    
    if loss<min_loss:
        torch.save(model, "./models/SimCLR_encoder.pt")
        
    print('Epoch: {}, loss: {}'.format(epoch, loss))
    history_train_loss.append(loss.data.item())


# save model
torch.save(model, "./models/SimCLR_ResNet_batch128_tao100.pt")