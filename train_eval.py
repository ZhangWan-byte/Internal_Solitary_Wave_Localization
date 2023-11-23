import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from dataset import *
from model import *
from metric import *
from loss_func import *

name_list = ["MLP", "OneDCNN", "EquiOneDCNN", "EquiResNet"]

all_model_names = ["MLP", "ResNet", "BoTNet", "OneDCNN", "EquiOneDCNN", "EquiResNet", "ConvNeXt", "EfficientNetv2"]

def training(X_train, y_train, X_val, y_val, batch_size=1024, lr=1e-4, epochs=400, model_name="MLP", loss_func="CE", gamma=2, input_length=96, dropout=0, path="", pretrain_path=None):
    
    # dataloader
    if model_name in name_list:
        alter_channel = False
    else:
        alter_channel = True

    train = myDataset(X_train, y_train, alter_channel=alter_channel)
    train_loader = torch.utils.data.DataLoader(dataset = train, batch_size = batch_size)
    val = myDataset(X_val, y_val, alter_channel=alter_channel)
    val_loader = torch.utils.data.DataLoader(dataset = val, batch_size = batch_size)

    device = torch.device("cuda")

    # model
    assert model_name in all_model_names
    
    if pretrain_path==None:
        if model_name == "MLP":
            model = MLP(num_class=17, input_length=input_length, dropout=dropout).to(device)
        elif model_name == "ResNet":
            model = get_ResNet(num_blocks=[3,4,6,3], num_classes=17, resolution=(16, 16)).to(device)
        elif model_name == "BoTNet":
            model = get_BoTNet(num_blocks=[3,4,6,3], num_classes=17, resolution=(16, 16), heads=4).to(device)
        elif model_name == "OneDCNN":
            model = OneDCNN(n_channels=64, hidden=128, n_classes=17).to(device)
        elif model_name == "EquiOneDCNN":
            model = EquiOneDCNN(n_channels=128, hidden=128, n_classes=17).to(device)
        elif model_name == "EquiResNet":
            model = RRPlus_M34res(n_channels=48, n_classes=17, eps=2e-5, use_bias=False).to(device)
        elif model_name == "ConvNeXt":
            model = ConvNeXt(in_chans=1, num_classes=17, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]).to(device)
        elif model_name == "EfficientNetv2":
            model = effnetv2_s(num_classes=17).to(device)
        else:
            print("something wrong happened!")
            exit()
    else:
        pretrain_model = torch.load(pretrain_path)
        model = ResNet_simclr(encoder=pretrain_model.encoder, num_classes=17).to(device)
            
    # loss and optimizer
    if loss_func == "CE":
        criterion = nn.CrossEntropyLoss()
    elif loss_func=="weightedLoss":
        y = np.array(y_train)
        weights = torch.tensor([len(np.where(y==i)[0]) for i in range(17)])
        weights = weights / weights.sum()
        weights = 1 / weights
        weights = weights / weights.sum()
        weights = weights.to(device)
        
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif loss_func == "FocalLoss":
        y = np.array(y_train)
        weights = torch.tensor([len(np.where(y==i)[0]) for i in range(17)])
        weights = weights / weights.sum()
        weights = 1 / weights
        weights = weights / weights.sum()
        weights = weights.to(device)
        
        criterion = FocalLoss3(alpha=weights,gamma=gamma)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    # training
    model.train()

    history_train_loss = []
    history_val_loss = []

    max_gmean = 0
    best_epoch = 0

    for epoch in range(epochs):
        """training"""
        for data in train_loader:
            X, y = data
            X, y = Variable(X), Variable(y)
            X, y = X.to(torch.float32), y.to(torch.float32)
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        """validating"""
        model.eval()
        val_loss = 0
        y_pred = []

        for data, target in val_loader:
            with torch.no_grad():
                data, target = Variable(data),Variable(target)
                data, target = data.to(device).to(torch.float32), target.to(device).to(torch.float32)
                output = model(data)
            y_pred.append(output.cpu().numpy())
            val_loss += criterion(output, target).data.item()
        val_loss /= len(val_loader)

        y_pred = np.vstack(y_pred)
        y_pred = to_single(y_pred)
        gmean = imblearn.metrics.geometric_mean_score(y_pred=y_pred, y_true=y_val, average='macro')

        if gmean > max_gmean:
            torch.save(model, "./results/{}.pt".format(model_name))
            max_gmean = gmean
            best_epoch = epoch

        print('Epoch: {}, loss: {}, val_loss: {}, gmean: {}, best_epoch: {}'.format(epoch, loss, val_loss, gmean, best_epoch))
        history_train_loss.append(loss.data.item())
        history_val_loss.append(val_loss)
        
    return history_train_loss, history_val_loss


def evaluating(history_train_loss, history_val_loss, X_test, y_test, model_path="./results/OneDCNN.pt", batch_size=1024, epochs=50, model_name="MLP", is_kfold=False, oversampling="", i=0, loss_func="CE"):
    if history_train_loss!=None and history_val_loss!=None:
        plt.title("loss curve")
        plt.plot(np.arange(epochs), history_train_loss)
        plt.plot(np.arange(epochs), history_val_loss)
        plt.legend(['train loss','val loss'])
        if is_kfold==True:
            is_oversampling = "_noOversampling" if oversampling=="" else "_"+oversampling
            loss = "" if loss_func=="CE" else "_"+loss_func
            plt.savefig("./data/imgs/"+"loss_"+model_name+is_oversampling+"_"+str(i)+loss)
        # plt.show()
    
    model = torch.load(model_path)
    
    device = torch.device("cuda")
    
    if model_name in name_list:
        alter_channel = False
    else:
        alter_channel = True

    test = myDataset(X_test, y_test, alter_channel=alter_channel)
    test_loader = torch.utils.data.DataLoader(dataset = test, batch_size = batch_size)
    
    model.eval()
    test_loss = 0
    y_pred = []

    for data, target in test_loader:
        with torch.no_grad():
            data, target = Variable(data),Variable(target)
            data, target = data.to(device).to(torch.float32), target.to(device).to(torch.float32)
            output = model(data)
        y_pred.append(output.cpu().numpy())

    y_pred = np.vstack(y_pred)
    y_pred = torch.tensor(y_pred)
    y_pred = nn.Softmax(dim=1)(y_pred).numpy()

    if is_kfold==True:
#         metric_kfold(y_pred=y_pred, y_test=y_test, model_name=model_name, i=i, oversampling=oversampling)
        return y_pred
    else:
        metric(y_pred=y_pred, y_test=y_test)