import torch
import torch.nn as nn
import torch.nn.functional as F

import eerie


class RRPlus_IdentityBlock(torch.nn.Module):
    def __init__(self, group, in_channels, out_channels, kernel_size, h_grid, h_grid_crop, stride, use_bias, eps, override=False):
        super(RRPlus_IdentityBlock, self).__init__()

        # Additional params
        self.diff_size = (in_channels != out_channels)
        self.h_grid = h_grid_crop if self.diff_size else h_grid
        self.crop = h_grid_crop.N
        self.override = override

        # Conv Layers
        if not override:
            self.c1 = eerie.nn.GConvGG(
                group, 
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                h_grid=self.h_grid, 
                stride=stride, 
                bias=use_bias, 
                h_crop=self.diff_size
            )
        else:
            self.c1 = eerie.nn.GConvGG(
                group, 
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                h_grid=h_grid_crop, 
                stride=stride, 
                bias=use_bias, h_crop=True
            )

        self.c2 = eerie.nn.GConvGG(
            group, 
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            h_grid=h_grid, 
            stride=stride, 
            bias=use_bias, 
            h_crop=False
        )

        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm2d(num_features=out_channels, eps=eps)
        self.bn2 = torch.nn.BatchNorm2d(num_features=out_channels, eps=eps)
        self.bn_out = torch.nn.BatchNorm2d(num_features=out_channels, eps=eps)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:])) ** (-1 / 2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        out = torch.relu(self.bn1(self.c1(x)))
        out = self.bn2(self.c2(out))
        # shortcut
        if self.diff_size:
            out = out + x[:,:,:-(self.crop - 1),:].repeat(1, 2, 1, 1)
        elif self.override:
            out = out + x[:, :, :-(self.crop - 1), :]
        else:
            out = out + x
        out = torch.relu(self.bn_out(out))
        return out


class RRPlus_M34res(torch.nn.Module):
    def __init__(self, n_channels=48, n_classes=10, eps=2e-5, use_bias=False):
        super(RRPlus_M34res, self).__init__()

        self.n_classes = n_classes

        # G-conv approach
        group = eerie.Group('R1R+')

        # For first layer:
        N_h_RdG = 9
        base = 2
        h_grid_RdG = group.h_grid_global(N_h_RdG, base ** (N_h_RdG - 1))
        print(h_grid_RdG.grid)

        # For subsequent layers:
        N_h = 1
        base = 2
        h_grid = group.h_grid_global(N_h, base ** (N_h - 1))
        print(h_grid.grid)
        n_channels_G = int(45)  # For Nh=6 use 2.45, For Nh = 7, use 2.65, For Nh=8, use 2.85, For Nh=9, use 3.0,

        N_h_crop = 3  # <--- TODO: not sure if this is the most optimal though, but it reduces the h_axis nicely to size 1 in the last layer
        base = 2
        h_grid_crop = group.h_grid_global(N_h_crop, base ** (N_h_crop - 1))
        print(h_grid_crop.grid)

        # Conv Layers
        self.c1 = eerie.nn.GConvRdG(
            group, 
            in_channels=6, 
            out_channels=n_channels_G, 
            kernel_size=7, 
            h_grid=h_grid_RdG, 
            bias=use_bias, 
            stride=1
        )
        # ----
        first_block = []
        for i in range(3):
            override = True if i == 0 else False
            first_block.append(
                RRPlus_IdentityBlock(
                    group, 
                    in_channels=n_channels_G, 
                    out_channels=n_channels_G, 
                    kernel_size=3, 
                    stride=1, 
                    h_grid=h_grid, 
                    h_grid_crop=h_grid_crop, 
                    use_bias=use_bias, 
                    eps=eps, 
                    override=override
                )
            )
        self.first_block = nn.Sequential(*first_block)
        # ----
        sec_block = []
        for i in range(4):
            b_channels = n_channels_G if i == 0 else n_channels_G * 2
            sec_block.append(
                RRPlus_IdentityBlock(
                    group, 
                    in_channels=b_channels, 
                    out_channels=n_channels_G * 2, 
                    kernel_size=3, 
                    stride=1, 
                    h_grid=h_grid, 
                    h_grid_crop=h_grid_crop, 
                    use_bias=use_bias, 
                    eps=eps
                )
            )
        self.sec_block = nn.Sequential(*sec_block)
        # ----
        thrd_block = []
        for i in range(6):
            b_channels = n_channels_G * 2 if i == 0 else n_channels_G * 4
            thrd_block.append(
                RRPlus_IdentityBlock(
                    group, 
                    in_channels=b_channels, 
                    out_channels=n_channels_G * 4, 
                    kernel_size=3, 
                    stride=1, 
                    h_grid=h_grid, 
                    h_grid_crop=h_grid_crop, 
                    use_bias=use_bias, 
                    eps=eps
                )
            )
        self.thrd_block = nn.Sequential(*thrd_block)
        # ----
        frth_block = []
        for i in range(3):
            b_channels = n_channels_G * 4 if i == 0 else n_channels_G * 8
            frth_block.append(
                RRPlus_IdentityBlock(
                    group, 
                    in_channels=b_channels, 
                    out_channels=n_channels_G * 8, 
                    kernel_size=3, 
                    stride=1, 
                    h_grid=h_grid, 
                    h_grid_crop=h_grid_crop, 
                    use_bias=use_bias, 
                    eps=eps
                )
            )
        self.frth_block = nn.Sequential(*frth_block)
        # ----
        self.c_out = eerie.nn.GConvGG(
            group, 
            in_channels=n_channels_G * 8, 
            out_channels=n_classes, 
            kernel_size=1, 
            stride=1, 
            h_grid=h_grid, 
            bias=use_bias)

        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm2d(num_features=n_channels_G, eps=eps)

        # Pooling
        self.pool = eerie.functional.max_pooling_R1

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        # Instead of strided conv, we use normal conv and then pooling.
        out = self.c1(x)
        out = self.pool(out, kernel_size=4, stride=1, padding=1)
        out = torch.relu(self.bn1(out))
        # print("0 ", out.shape)
        # -----
        out = self.pool(out, kernel_size=4, stride=1, padding=1)
        out = self.first_block(out)
        # print("1 ", out.shape)
        out = self.pool(out, kernel_size=4, stride=1, padding=1)
        out = self.sec_block(out)
        # print("2 ", out.shape)
        out = self.pool(out, kernel_size=4, stride=1, padding=1)
        out = self.thrd_block(out)
        # print("3 ", out.shape)
        out = self.pool(out, kernel_size=4, stride=4, padding=1)
        out = self.frth_block(out)
        # print("4 ", out.shape)
        # Global pooling
        out = torch.mean(out, dim=-1, keepdim=True) # pool over the time axis
        out = self.c_out(out)
        # print("5 ", out.shape)
        # Then turn into features per time point (merging scale and the channel axes)
        out = torch.max(out, dim=-2).values  # pool over the scale axis
        out = out.view(out.size(0), self.n_classes)
        return out