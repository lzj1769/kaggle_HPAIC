from __future__ import print_function
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from configure import *

df = pd.read_csv("/home/rs619065/kaggle_HPAIC/training/PreTrained_ResNet_50_LogLoss_train_f1_0.808_val_f1_0.616.csv")
y_true = list()
y_pred = list()

mlb = MultiLabelBinarizer(classes=range(N_LABELS))

for i in range(df.shape[0]):
    y_true.append(map(int, df['Target'][i].split(" ")))
    y_pred.append(map(int, df['Predicted'][i].split(" ")))

y_true = mlb.fit_transform(y_true)
y_pred = mlb.fit_transform(y_pred)


for i in range(N_LABELS):
    f1 = f1_score(y_true[:, i], y_pred[:, i])
    print(list(set(y_true[:, i])))
    print(list(set(y_pred[:, i])))
    print("Class: {}, F1: {}".format(i, f1))

print(f1_score(y_true, y_pred, average="macro"))