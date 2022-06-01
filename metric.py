import imblearn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, matthews_corrcoef


def to_single(y_pred):
    li = []
    
    for i in y_pred:
        li.append(np.argmax(i))
    
    return np.array(li)


def fuzzy_acc(y_true, y_pred, err=1):
    true = 0
    false = 0
    for i in range(len(y_true)):
        if y_pred[i]==0 and y_true[i]==0:
            true += 1
        elif y_pred[i]!=0 and y_true[i]!=0 and abs(y_pred[i]-y_true[i])<=err:
            true += 1
        else:
            false += 1
    acc = true/(true+false)
    return acc

    
def metric_kfold(y_pred, y_test, model_name, i, oversampling, loss_func="CE"):
    
    oversampling = "_noOversampling" if oversampling=="" else "_"+oversampling
    
    y_pred_proba = y_pred
    if (np.sum(y_pred[0]) - 1) >= 1e-5:
        y_pred_proba = nn.Softmax()(y_pred_proba)
    
    y_pred = to_single(y_pred_proba)
    
    C = confusion_matrix(y_pred=y_pred, y_true=y_test)
    cmd = ConfusionMatrixDisplay(confusion_matrix=C)
    fig, ax = plt.subplots(figsize=(12,12))
    cmd.plot(ax=ax,cmap="Greens")
    loss = "" if loss_func=="CE" else "_"+loss_func
    plt.savefig("./data/imgs/"+model_name+oversampling+"_"+str(i)+loss)
    # plt.show()
    
    Acc = round(accuracy_score(y_pred=y_pred, y_true=y_test), 6)
    Pre = round(precision_score(y_pred=y_pred, y_true=y_test, average='macro'), 6)
    Rec = round(recall_score(y_pred=y_pred, y_true=y_test, average='macro'), 6)
    F1 = round(f1_score(y_pred=y_pred, y_true=y_test, average='macro'), 6)
    G_mean_micro = round(imblearn.metrics.geometric_mean_score(y_pred=y_pred, y_true=y_test, average='micro'), 6)
    G_mean_macro = round(imblearn.metrics.geometric_mean_score(y_pred=y_pred, y_true=y_test, average='macro'), 6)
    MAUC = round(roc_auc_score(y_true=y_test, y_score=y_pred_proba, multi_class="ovo"), 6)
    MMCC = round(matthews_corrcoef(y_true=y_test, y_pred=y_pred), 6)
    Acc_1 = round(fuzzy_acc(y_true=y_test, y_pred=y_pred, err=1), 6)
    Acc_3 = round(fuzzy_acc(y_true=y_test, y_pred=y_pred, err=3), 6)
    Acc_5 = round(fuzzy_acc(y_true=y_test, y_pred=y_pred, err=5), 6)
    
    return [Acc, Pre, Rec, F1, G_mean_micro, G_mean_macro, MAUC, MMCC, Acc_1, Acc_3, Acc_5]