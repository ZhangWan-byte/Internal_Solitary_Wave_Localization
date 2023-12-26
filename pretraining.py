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
from simclr.modules.identity import Identity
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

import argparse
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
                # torchvision.transforms.RandomApply([color_jitter], p=0.8),
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


class mySimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, n_features=17, hidden_features=2048, projection_dim=64):
        super(mySimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features
        self.hidden_features = hidden_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.hidden_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='num of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--world_size', type=int, default=1)

    parser.add_argument('--hidden_features', type=int, default=2048)
    parser.add_argument('--projection_dim', type=int, default=64)

    parser.add_argument('--augment', type=int, default=1, help='whether unsupervised data are flipping augmented')
    args = parser.parse_args()

    # load global test data (unlabeled data)
    path = "./data/global_southSea_data_augment.npy" if args.augment==1 else "./data/global_southSea_data.npy"
    global_test = np.load(path)
    print(global_test.shape)


    # epochs = 1#10
    # batch_size = 128 #256 #512 #1024
    # lr = 3e-4
    # temperature = 0.1 # 100 # 0.1
    # world_size = 1

    device = torch.device("cuda")

    transform = myTransformsSimCLR(size=[6,16])
    train = myDataset(data=global_test, label=None, is_global_test=True, alter_channel=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset = train, batch_size=args.batch_size)

    encoder = RRPlus_M34res(n_channels=48, n_classes=17, eps=2e-5, use_bias=False).cuda() # ResNet50() # BoTNet50()
    # n_features = 2048 # get dimensions of last fully-connected layer
    # projection_dim = 64
    model = mySimCLR(encoder, 17, args.hidden_features, args.projection_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

    history_train_loss = []
    min_loss = 100
    model.train()

    for epoch in range(args.epochs):
        for (x_i, x_j), _ in tqdm(train_loader):
            if x_i.shape[0]!=args.batch_size:
                break
            optimizer.zero_grad()
            x_i = x_i.squeeze().cuda(non_blocking=True)
            x_j = x_j.squeeze().cuda(non_blocking=True) # non_blocking=True

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
    torch.save(model, "./models/SimCLR_EquiResNet_batch{}_tau{}_augment{}.pt".format(args.batch_size, args.temperature, args.augment))