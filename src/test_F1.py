from __future__ import print_function
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from configure import *

df = pd.read_csv("/home/rs619065/kaggle_HPAIC/validation/PreTrained_MobileNet_LogLoss_train_f1_0.804_val_f1_0.618.csv")
y_true = list()
y_pred = list()

mlb = MultiLabelBinarizer(classes=range(N_LABELS))

for i in range(df.shape[0]):
    y_true.append(map(int, df['Target'][i].split(" ")))
    y_pred.append(map(int, df['Predicted'][i].split(" ")))

print(y_true[0])
y_true = mlb.fit_transform(y_true)
y_pred = mlb.fit_transform(y_pred)
print(y_true)
print(y_pred)

f1 = f1_score(y_true, y_pred, average="macro").round(3)
print(f1)