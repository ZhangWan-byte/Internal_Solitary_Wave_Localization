import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import warnings
import joblib
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from dataset import *
from metric import *
from model import *
from train_eval import *
from loss_func import *

warnings.filterwarnings('ignore')


def kfold_training(model_name, kfold=2, times=5, oversampling="", data_shape='1x96', loss_func="CE", lr=3e-4, epoch=200, pretrain_path=None, batch_size=1024):
    
    metric_li = []
    metric_names = ["Acc", "Pre", "Rec", "F1", "G_mean_micro", "G_mean_macro", "MAUC", "MMCC", "Acc_1", "Acc_3", "Acc_5"]
    
    for t in range(times):
        for i in range(kfold):
            X_train, y_train, X_test, y_test = load_data_kfold(
                data_shape=data_shape, oversampling=oversampling, is_augment=False, kfold=kfold, i=i, t=t)

            if model_name=="RF":
                rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
                rf.fit(X_train, y_train)
                y_pred = rf.predict_proba(X_test)

            elif model_name=="LGB":
                lgb = LGBMClassifier(n_estimators=50, max_depth=10, random_state=42)
                lgb.fit(X_train, y_train)
                y_pred = lgb.predict_proba(X_test)

            elif model_name=="MLP":
                history_train_loss, history_val_loss = training(
                    X_train, y_train, X_test, y_test, 
                    batch_size=batch_size, lr=lr, epochs=200, model_name=model_name, 
                    loss_func=loss_func)

                y_pred = evaluating(
                    history_train_loss, history_val_loss, X_test, y_test, 
                    model_path="./results/{}.pt".format(model_name), batch_size=batch_size, epochs=200, 
                    model_name=model_name, is_kfold=True, oversampling=oversampling, i=i, loss_func="CE")

            elif model_name=="ResNet":
                history_train_loss, history_val_loss = training(
                    X_train, y_train, X_test, y_test, 
                    batch_size=batch_size, lr=lr, epochs=epoch, model_name=model_name, 
                    loss_func=loss_func, pretrain_path=pretrain_path)

                y_pred = evaluating(
                    history_train_loss, history_val_loss, X_test, y_test, 
                    model_path="./results/{}.pt".format(model_name), batch_size=batch_size, epochs=epoch, 
                    model_name=model_name, is_kfold=True, oversampling=oversampling, i=i, loss_func="CE")

            elif model_name=="BoTNet":
                history_train_loss, history_val_loss = training(
                    X_train, y_train, X_test, y_test, 
                    batch_size=batch_size, lr=lr, epochs=epoch, model_name=model_name, 
                    loss_func=loss_func, pretrain_path=pretrain_path)

                y_pred = evaluating(
                    history_train_loss, history_val_loss, X_test, y_test, 
                    model_path="./results/{}.pt".format(model_name), batch_size=batch_size, epochs=epoch, 
                    model_name=model_name, is_kfold=True, oversampling=oversampling, i=i, loss_func="CE")

            elif model_name=="ResNet_pretrain":
                pass
            elif model_name=="BoTNet_pretrain":
                pass

            elif model_name=="OneDCNN":
                history_train_loss, history_val_loss = training(
                    X_train, y_train, X_test, y_test, 
                    batch_size=batch_size, lr=lr, epochs=epoch, model_name=model_name, 
                    loss_func=loss_func, pretrain_path=pretrain_path)

                y_pred = evaluating(
                    history_train_loss, history_val_loss, X_test, y_test, 
                    model_path="./results/{}.pt".format(model_name), batch_size=batch_size, epochs=epoch, 
                    model_name=model_name, is_kfold=True, oversampling=oversampling, i=i, loss_func="CE")

            elif model_name=="EquiOneDCNN":
                history_train_loss, history_val_loss = training(
                    X_train, y_train, X_test, y_test, 
                    batch_size=batch_size, lr=lr, epochs=epoch, model_name=model_name, 
                    loss_func=loss_func, pretrain_path=pretrain_path)

                y_pred = evaluating(
                    history_train_loss, history_val_loss, X_test, y_test, 
                    model_path="./results/{}.pt".format(model_name), batch_size=batch_size, epochs=epoch, 
                    model_name=model_name, is_kfold=True, oversampling=oversampling, i=i, loss_func="CE")

            metric_li.append(metric_kfold(y_pred, y_test, model_name=model_name, i=i, oversampling=oversampling))
            print(metric_li[-1])

    m = np.vstack(metric_li)
    mean = m.mean(axis=0)
    std = m.std(axis=0)

    for i in range(len(metric_names)):
        print("{}\t\t{}±{}".format(metric_names[i], round(mean[i], 6), round(std[i], 6)))

    str_oversample = "noOversampling" if oversampling=="" else oversampling
    str_pretrain_path = "noPretrain" if pretrain_path==None else pretrain_path.split("/")[-1].split(".")[0]
    np.save("./data/{}fold/metric/{}_{}_{}_{}_{}_{}_{}_{}".format(kfold,
                                                                  str_oversample,
                                                                  model_name,
                                                                  data_shape,
                                                                  loss_func,
                                                                  str(lr),
                                                                  str(epoch),
                                                                  str(batch_size),
                                                                  str_pretrain_path), m)


# code demos are as following:

# RF/LGB models
# kfold_training(model_name="RF", oversampling="", data_shape="1x96")
# kfold_training(model_name="RF", oversampling="oversample", data_shape="1x96")

# Neural Network models
# kfold_training(model_name="MLP", oversampling="", data_shape="1x96")
# kfold_training(model_name="MLP", oversampling="SMOTE", data_shape="1x96")
# kfold_training(model_name="ResNet", oversampling="", data_shape="16x16x1", lr=3e-4, epoch=400)
kfold_training(model_name="OneDCNN", oversampling="", data_shape="6x16", lr=3e-4, epoch=400)
# kfold_training(model_name="EquiOneDCNN", oversampling="", data_shape="6x16", lr=3e-4, epoch=400)

# Using pre-training
# kfold_training(model_name="BoTNet", oversampling="", data_shape='16x16x1', loss_func="FocalLoss", lr=6e-4, epoch=400, pretrain_path="./results/SimCLR_BoTNet_southSea_batch1024_proj64_tao100_lr1e-3_10epoch.pt", batch_size=128)
