from __future__ import print_function
from __future__ import division

import os
import glob
import math
from collections import Counter
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from configure import *

NET_NAMES = ['ResNet50', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'InceptionV3', 'InceptionResNetV2', 'LightGBM', 'Average']


def get_data():
    df_train = None
    df_test = None

    for net_name in NET_NAMES:
        filename = glob.glob(os.path.join(TRAINING_OUTPUT_PATH, net_name, "{}_f1_*.csv".format(net_name)))[0]
        df = pd.read_csv(filename)

        if df_train is None:
            df_train = df
            df_train.rename(columns={'Predicted': net_name}, inplace=True)

        else:
            df_train[net_name] = df['Predicted']

        filename = glob.glob(os.path.join(SUBMISSION_PATH, net_name, "{}_val_f1_*.csv".format(net_name)))[0]
        df = pd.read_csv(filename)

        if df_test is None:
            df_test = df
            df_test.rename(columns={'Predicted': net_name}, inplace=True)

        else:
            df_test[net_name] = df['Predicted']

    return df_train, df_test


def voting(df):
    voting_res = list()

    for i in range(df.shape[0]):
        pred_label = list()
        voting = list()

        for net_name in NET_NAMES:
            pred_label += df[net_name][i].split()

        counts = Counter(pred_label)

        for key in counts.keys():
            count = counts[key]
            if count >= int(len(NET_NAMES) / 2.0):
                voting.append(key)

        # if majority class cannot be found
        if len(voting) == 0:
            voting += counts.keys()

        voting_res.append(" ".join(voting))

    df['Predicted'] = voting_res

    return df


def main():
    df_train, df_test = get_data()

    df_train = voting(df_train)
    df_test = voting(df_test)

    mlb = MultiLabelBinarizer(classes=range(N_LABELS))
    y_true = list()
    y_pred = list()
    for i in range(df_train.shape[0]):
        y_true.append(map(int, df_train['Target'][i].split(" ")))
        y_pred.append(map(int, df_train['Predicted'][i].split(" ")))

    f1 = f1_score(y_true=mlb.fit_transform(y_true),
                  y_pred=mlb.fit_transform(y_pred),
                  average='macro')

    print(f1)
    filename = os.path.join("/home/rs619065/HPAIC/training/MajorityVoting", "MajorityVoting_f1_{}.csv".format(f1))
    df_train.to_csv(filename, index=False)

    filename = os.path.join("/home/rs619065/HPAIC/submission/MajorityVoting", "MajorityVoting_f1_{}.csv".format(f1))
    df_test.to_csv(filename, index=False)

    df_test_sub = df_test[['Id', 'Predicted']]
    filename = os.path.join("/home/rs619065/HPAIC/submission/MajorityVoting", "MajorityVoting_f1_{}_submission.csv".format(f1))
    df_test_sub.to_csv(filename, index=False)


if __name__ == '__main__':
    main()