# adapted from https://github.com/dwromero/wavelet_networks
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import eerie

class OneDCNN(torch.nn.Module):
    def __init__(self, n_channels=16, hidden=16, n_classes=17, eps=2e-5, use_bias=False, dp_rate=0.25):
        super(OneDCNN, self).__init__()

        # Conv Layers
        self.c1 = torch.nn.Conv1d(in_channels=6,              out_channels=n_channels,      kernel_size=8,  stride=1, padding='same', dilation=1, bias=use_bias)
        self.c2 = torch.nn.Conv1d(in_channels=n_channels,     out_channels=n_channels * 2,  kernel_size=4,  stride=1, padding='same', dilation=1, bias=use_bias)
        self.c3 = torch.nn.Conv1d(in_channels=n_channels * 2, out_channels=n_channels * 4,  kernel_size=2,  stride=1, padding='same', dilation=1, bias=use_bias)
        
        # Fully connected
        self.f1 = torch.nn.Linear(in_features=n_channels * 4, out_features=hidden,    bias=True)
        self.f2 = torch.nn.Linear(in_features=hidden,         out_features=hidden,    bias=True)
        self.f3 = torch.nn.Linear(in_features=hidden,         out_features=n_classes, bias=True)

        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm1d(num_features=n_channels,      eps=eps)
        self.bn2 = torch.nn.BatchNorm1d(num_features=n_channels * 2,  eps=eps)
        self.bn3 = torch.nn.BatchNorm1d(num_features=n_channels * 4,  eps=eps)

        # Pooling
        self.pool = torch.max_pool1d
        # self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding='same')

        # DropOut
        self.dropout = torch.nn.Dropout(p=dp_rate)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        # Conv-layers
        out = self.bn1(torch.relu(self.c1(x)))
        # print("1 ", out.shape)
        # out = self.pool(out, kernel_size=8, stride=8, padding=0)
        out = self.pool(out, kernel_size=4, stride=2, padding=1)
        # print("1 ", out.shape)
        out = self.bn2(torch.relu(self.c2(out)))
        # print("2 ", out.shape)
        out = self.pool(out, kernel_size=4, stride=2, padding=1)
        # print("2 ", out.shape)
        out = self.bn3(torch.relu(self.c3(out)))
        # print("3 ", out.shape)
        out = self.pool(out, kernel_size=4, stride=4, padding=1)
        # print("3 ", out.shape)

        # Fully connected lyrs
        out = out.view(out.size(0), -1)
        out = self.dropout(self.f1(out))
        out = self.dropout(self.f2(out))
        out = self.f3(out)

        return out


class EquiOneDCNN(torch.nn.Module):
    def __init__(self, n_channels=16, hidden=16, n_classes=17, eps=2e-5, use_bias=False, dp_rate=0.25):
        super(EquiOneDCNN, self).__init__()

        # # G-conv approach
        group = eerie.Group('R1R+')

        # For first layer:
        N_h_RdG = 9
        base = 2
        h_grid_RdG = group.h_grid_global(N_h_RdG, base ** (N_h_RdG - 1))
        # print(h_grid_RdG.grid)

        N_h_crop = 3  # <--- TODO: not sure if this is the most optimal though, but it reduces the h_axis nicely to size 1 in the last layer
        base = 2
        h_grid_crop = group.h_grid_global(N_h_crop, base ** (N_h_crop - 1))
        # print(h_grid_crop.grid)

        # Conv Layers
        self.c1 = eerie.nn.GConvRdG(group, in_channels=6,              out_channels=n_channels,     kernel_size=7, h_grid=h_grid_RdG,  bias=use_bias, stride=1)
        self.c2 = eerie.nn.GConvGG(group,  in_channels=n_channels,     out_channels=n_channels * 2, kernel_size=3, h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.c3 = eerie.nn.GConvGG(group,  in_channels=n_channels * 2, out_channels=n_channels * 4, kernel_size=1, h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        
        # Fully connected
        self.f1 = torch.nn.Linear(in_features=n_channels * 4 * (N_h_RdG+1)//2, out_features=hidden, bias=True)
        self.f2 = torch.nn.Linear(in_features=hidden, out_features=hidden,    bias=True)
        self.f3 = torch.nn.Linear(in_features=hidden, out_features=n_classes, bias=True)

        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm2d(num_features=n_channels,      eps=eps)
        self.bn2 = torch.nn.BatchNorm2d(num_features=n_channels * 2,  eps=eps)
        self.bn3 = torch.nn.BatchNorm2d(num_features=n_channels * 4,  eps=eps)
        
        # Pooling
        self.pool = eerie.functional.max_pooling_R1
        # DropOut
        self.dropout = torch.nn.Dropout(p=dp_rate)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        # Conv-layers
        # We replace strided convolutions with normal convolutions followed by max pooling.
        # -----
        out = self.c1(x)
        # print("1 ", out.shape)
        out = self.pool(out, kernel_size=4, stride=2, padding=1)
        out = self.bn1(torch.relu(out))
        # print("1 ", out.shape)
        # -----
        # out = self.pool(out, kernel_size=8, stride=8, padding=0)
        # print("1 ", out.shape)
        # -----
        out = self.c2(out)
        # print("2 ", out.shape)
        out = self.pool(out, kernel_size=4, stride=2, padding=1)
        out = self.bn2(torch.relu(out))
        # print("2 ", out.shape)
        # -----
        # out = self.pool(out, kernel_size=8, stride=8, padding=0)
        # print("2 ", out.shape)
        # -----
        out = self.c3(out)
        # print("3 ", out.shape)
        out = self.pool(out, kernel_size=4, stride=4, padding=1)
        out = self.bn3(torch.relu(out))
        # print("3 ", out.shape)
        # -----
        # Fully connected lyrs
        out = out.view(out.size(0), -1)
        out = self.dropout(self.f1(out))
        out = self.dropout(self.f2(out))
        out = self.f3(out)

        return out