import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model


def load_data(data_shape='1x96', normalization='standard',oversampling="", is_global_test=False, is_augment=False, only_south_sea=False,path=""):
    
    if is_global_test==True:
        if only_south_sea==False:
            global_test = np.load(".{}/data/test/test_data_{}.npy".format(path, data_shape))
        else:
            global_test = np.load(".{}/data/test/test_data_{}_southSea.npy".format(path, data_shape))
    
        return global_test
    
    elif is_global_test==False:
        oversample = "" if oversampling=="" else oversampling+"_"
        augment = "nonAugment_" if is_augment==False else ""

        X_train = np.load(".{}/data/{}/{}{}{}_X_train.npy".format(path, data_shape, augment, oversample, normalization))
        y_train = np.load(".{}/data/{}/{}{}{}_y_train.npy".format(path, data_shape, augment, oversample, normalization))
        X_val = np.load(".{}/data/{}/{}{}{}_X_val.npy".format(path, data_shape, augment, oversample, normalization))
        y_val = np.load(".{}/data/{}/{}{}{}_y_val.npy".format(path, data_shape, augment, oversample, normalization))
        X_test = np.load(".{}/data/{}/{}{}{}_X_test.npy".format(path, data_shape, augment, oversample, normalization))
        y_test = np.load(".{}/data/{}/{}{}{}_y_test.npy".format(path, data_shape, augment, oversample, normalization))

        print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

        return X_train, y_train, X_val, y_val, X_test, y_test
    
def load_data_kfold(data_shape='1x96', oversampling="", is_augment=False, i=1, kfold=2,t=1):
    
    is_oversampling = "noOversampling" if oversampling=="" else oversampling
    is_augment = "" if is_augment==False else "_augmented"

    X_train = np.load("./data/"+str(kfold)+"fold/"+data_shape+"/T"+str(t)+"_fold"+str(i)+"_"+is_oversampling+"_X_train"+is_augment+".npy")
    y_train = np.load("./data/"+str(kfold)+"fold/"+data_shape+"/T"+str(t)+"_fold"+str(i)+"_"+is_oversampling+"_y_train"+is_augment+".npy")
    X_test = np.load("./data/"+str(kfold)+"fold/"+data_shape+"/T"+str(t)+"_fold"+str(i)+"_"+is_oversampling+"_X_test"+is_augment+".npy")
    y_test = np.load("./data/"+str(kfold)+"fold/"+data_shape+"/T"+str(t)+"_fold"+str(i)+"_"+is_oversampling+"_y_test"+is_augment+".npy")

    print("./data/"+str(kfold)+"fold/"+data_shape+"/T"+str(t)+"_fold"+str(i)+"_"+is_oversampling+"_X_train"+is_augment+".npy"+" loaded")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    return X_train, y_train, X_test, y_test
    
class myDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, is_global_test=False, alter_channel=False, transform=None):
        self.is_global_test = is_global_test
        self.alter_channel = alter_channel
        self.transform = transform
        
        if self.is_global_test==False:
            self.data = torch.from_numpy(data)
            self.label = torch.from_numpy(label)
        else:
            self.data = torch.from_numpy(data)
            self.label = None
            
    def __getitem__(self, index):
        if self.transform==None:
            if self.is_global_test==False:
                X, y = self.data[index], self.label[index]
                if self.alter_channel == True:
                    X = X.reshape(1,16,16)
                    X = np.vstack([X,X,X])
                y_li = [0]*17
                y_li[int(y.numpy())]=1
                return X, torch.tensor(y_li)
            else:
                X = self.data[index]
                return X, _
        else:
            if self.is_global_test==False:
                X = self.data[index]
                return X, _
            else:
                X = self.data[index]
                if self.alter_channel == True:
                    X = X.reshape(1,16,16)
                    X = np.vstack([X,X,X])
                    # X = torch.tensor(X).to(torch.float32)
                return self.transform(X), self.transform(X)
            
    def __len__(self):
        return len(self.data)
    