# This script amis to test the average f1 score

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from configure import *

df_validation = pd.read_csv(VALIDATION_DATA_CSV)

y_true = list()
y_pred = list()

for i in range(3):
    targets = map(int, df_validation.iloc[i][1].split(" "))
    labels = [0] * N_LABELS
    for target in targets:
        labels[target] = 1
    y_true.append(labels)
    y_pred.append(labels)

y_true = np.array([[1, 2, 3], [1, 2], [1, 3]])
y_pred = np.array([[3, 1, 2], [2, 1], [3, 1]])

mlb = MultiLabelBinarizer(classes=range(N_LABELS))

y_true = mlb.fit_transform(y_true)
y_pred = mlb.fit_transform(y_pred)

print(y_true.shape)
print(y_pred.shape)
f1 = f1_score(y_true=mlb.fit_transform(y_true), y_pred=mlb.fit_transform(y_pred), average="macro")
print("F1 score for perfect prediction is {}".format(f1))
