import scipy.io as scio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import joblib
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN

# slice (n,96) data into (m,16x16x1)
def to_size_16x16(X_train):
    Z = np.zeros((16,10))

    temp_li = []
    for i in range(int(X_train.shape[0])):
        temp = np.array(X_train.iloc[i, :])
        temp = temp.reshape(16, 6)
        temp = np.hstack((temp, Z))
        temp = temp.reshape((16,16,1))
        temp_li.append(temp)
    X_train = np.stack(temp_li,axis=0)
    
    return X_train

# slice (n,96) data into (m,6x16)
def to_size_6x16(X_train):
    
    temp_li = []
    for i in range(int(X_train.shape[0])):
        temp = np.array(X_train.iloc[i, :])
        temp = temp.reshape(16, 6)
        temp_li.append(temp)
    X_train = np.stack(temp_li,axis=0)
    
    X_train = X_train.transpose(0, 2, 1)             # (batch, 16, 6) -> (batch, 6, 16) each row is a feature variable

    return X_train


# keep data points in south china sea
def is_south_sea(df):
    is_in_south_sea = True
    
    for i in range(len(df)):
        if not (85<=df.iloc[i,0]<=130 and -10<=df.iloc[i,1]<=40):
            is_in_south_sea = False
            break
    return is_in_south_sea



def global_data_generation(data_path, is_augment=False, only_south_sea=True):
    
    # data loading
    data_inputfilenames = os.listdir(data_path)
    data_li = []
    for files in data_inputfilenames:
        readpath = data_path + files
        intemp = scio.loadmat(readpath)
        intemp1 = intemp['inputs']
        intemp1[np.isnan(intemp1)]=0
        data_li.append(intemp1)
    data = np.hstack(data_li)
    
    # augmentation
    if is_augment==True:
        data_reverse = np.fliplr(data)
        data = np.hstack([data, data_reverse])
        label_reverse = np.fliplr(label)
        label = np.hstack([label, label_reverse])
    
    # data transformation
    data = pd.DataFrame(data)
    data = data.T
    
    # preserve only south sea
    if only_south_sea==True:
        temp_li = []
        for i in tqdm(range(int(data.shape[0]/16))):
            temp = data.iloc[:16, :]
            if is_south_sea(temp.iloc[:16,:2])==True:
                temp_li.append(temp)
            data = data.iloc[16:, :]
        data = np.vstack(temp_li)
    
    data = pd.DataFrame(data)
    data = data.iloc[:, [2,3,4,5,8,9]]
    data.columns = ["sigma0","mss","swh","sla","wind_speed","month"]
    print(data.shape)
    print("before: ", data.iloc[0,0])
    print("mean and std: ", np.mean(data.iloc[:,0]), np.std(data.iloc[:,0]))
    data = data.iloc[:,:].apply(lambda x: (x-np.mean(x))/np.std(x))
    print("after: ", data.iloc[0,0])
    print(data.shape)

    
    # data transformation
    temp_li = []
    for i in tqdm(range(int(data.shape[0]/16))):
        temp = data.iloc[:16, :]
        temp = temp.values.reshape(-1)
        temp_li.append(temp)
        data = data.iloc[16:, :]
    data = np.stack(temp_li,axis=0)
    data = to_size_16x16(pd.DataFrame(data))  
        
    print(data.shape)

    np.save("./data/global_southSea_data.npy", data)

    return data


def kfold_load_process_save_data(data_path, label_path, is_augment=False, data_shape='1x96', oversampling="", kfold=2, \
                                 only_south_sea=False, times=5):
    print()
    print(data_shape, oversampling)
    is_oversampling = "noOversampling" if oversampling=="" else oversampling
    is_augment = "" if is_augment==False else "_augmented"
    
    # 1. label
    label_inputfilenames = os.listdir(label_path)
    label_li = []
    for files in label_inputfilenames:
        readpath = label_path + files
        intemp = scio.loadmat(readpath)
        intemp1 = intemp['outputs']
        intemp1[np.isnan(intemp1)]=0
        label_li.append(intemp1)
    label = np.hstack(label_li)
    
    
    
    # 1. data
    data_inputfilenames = os.listdir(data_path)
    data_li = []
    for files in data_inputfilenames:
        readpath = data_path + files
        intemp = scio.loadmat(readpath)
        intemp1 = intemp['inputs']
        intemp1[np.isnan(intemp1)]=0
        data_li.append(intemp1)
    data = np.hstack(data_li)
    
    # data augmentation
    if is_augment==True:
        data_reverse = np.fliplr(data)
        data = np.hstack([data, data_reverse])
        
        label_reverse = np.fliplr(label)
        label = np.hstack([label, label_reverse])
    
    
    
    # 2. label
    label = label[0]
    label_li = []
    for i in range(int(len(label)/16)):
        temp = label[:16]
        if sum(temp)==0:
            label_li.append(0)
        else:
            label_li.append(np.argmax(temp)+1)
        label = label[16:]
    label = np.array(label_li)

    
    
    # 2. data
    data = pd.DataFrame(data)
    data = data.T
    
    # preserve only south sea
    if only_south_sea==True:
        temp_li = []
        for i in tqdm(range(int(data.shape[0]/16))):
            temp = data.iloc[:16, :]
            if is_south_sea(temp.iloc[:16,:2])==True:
                temp_li.append(temp)
            data = data.iloc[16:, :]
        data = np.vstack(temp_li)
    
    data = pd.DataFrame(data)
    data = data.iloc[:, [2,3,4,5,8,9]]
    data.columns = ["sigma0","mss","swh","sla","wind_speed","month"]

    
    # concat data and label
    temp_li = []
    for i in tqdm(range(int(data.shape[0]/16))):
        temp = data.iloc[:16, :]
        temp = temp.values.reshape(-1)
        temp_li.append(temp)
        data = data.iloc[16:, :]
    data = np.stack(temp_li,axis=0)
        
    print(data.shape, label.shape)
    data_and_label = np.hstack([data, label.reshape(-1,1)])
    data_and_label = pd.DataFrame(data_and_label)
    
    
    
    for t in range(times):
        data_and_label = data_and_label.sample(frac=1)
    
        # split to k fold and normalization
        kfold_list = np.array_split(data_and_label, kfold)

        for i in range(kfold):

            # get train data
            i_list = [num for num in range(kfold)]
            for num in i_list:
                if num==i:
                    i_list.pop(num)

            train = pd.concat([kfold_list[j] for j in i_list])

            X_train = train.iloc[:,:-1].apply(lambda x: (x-np.mean(x))/np.std(x))
            y_train = train.iloc[:, -1]

            # get test data
            test = kfold_list[i]
            X_test = test.iloc[:,:-1].apply(lambda x: (x-np.mean(x))/np.std(x))
            y_test = test.iloc[:, -1]

            # reshape to 16x16x1 if needed
            if data_shape=='16x16x1':
                X_train = to_size_16x16(X_train)
                X_test = to_size_16x16(X_test)

            if data_shape=='6x16':
                X_train = to_size_6x16(X_train)
                X_test = to_size_6x16(X_test)

            # oversampling techniques
            if oversampling!="":
                if data_shape=='16x16x1':
                    X_train = X_train.reshape(X_train.shape[0], 16*16*1)

                if oversampling == "oversample":
                    sampler = RandomOverSampler(random_state=42)
                elif oversampling == "SMOTE":
                    sampler = SMOTE(random_state=42)
                elif oversampling == "BorderlineSMOTE":
                    sampler = BorderlineSMOTE(random_state=42)
                elif oversampling == "ADASYN":
                    sampler =  ADASYN(random_state=42)
                X_train, y_train = sampler.fit_resample(X_train, y_train)

                if data_shape=='16x16x1':
                    X_train = X_train.reshape(X_train.shape[0], 16, 16, 1)

            print("times", t, "fold", i)
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


            np.save("./data/"+str(kfold)+"fold/"+data_shape+"/T"+str(t)+"_fold"+str(i)+"_"+is_oversampling+"_X_train"+is_augment, X_train)
            np.save("./data/"+str(kfold)+"fold/"+data_shape+"/T"+str(t)+"_fold"+str(i)+"_"+is_oversampling+"_y_train"+is_augment, y_train)
            np.save("./data/"+str(kfold)+"fold/"+data_shape+"/T"+str(t)+"_fold"+str(i)+"_"+is_oversampling+"_X_test"+is_augment, X_test)
            np.save("./data/"+str(kfold)+"fold/"+data_shape+"/T"+str(t)+"_fold"+str(i)+"_"+is_oversampling+"_y_test"+is_augment, y_test)


if __name__ == "__main__":
    # generating dataset 
    data_path="./data/Trainingdata_WS_Month_1.0/inputs/"
    label_path="./data/Trainingdata_WS_Month_1.0/labels/"

    data_shape = ["1x96", "16x16x1", "6x16"]
    oversampling = ["", "oversample", "SMOTE", "BorderlineSMOTE", "ADASYN"]

    for i in data_shape:
        for j in oversampling:
            kfold_load_process_save_data(data_path, label_path, is_augment=False, data_shape=i, oversampling=j, kfold=2, times=5)

    # generating pretraining dataset 
    # path_data = "D:/Inner_Wave/version1.0/Testingdata_global_1.0/"
    # global_data_generation(path_data, is_augment=False, only_south_sea=True)