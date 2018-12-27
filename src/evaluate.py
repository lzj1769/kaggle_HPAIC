from __future__ import print_function
from __future__ import division

import os
import sys

import pandas as pd
import numpy as np
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize': (12, 10)})

from sklearn.metrics import f1_score, precision_recall_curve

from configure import *
from utils import get_submission_path, get_training_predict_path, get_test_predict_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_name", help='name of convolutional neural network', default=None)
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity mode. DEFAULT: 2")
    return parser.parse_args()


def search_threshold(y_true, y_pred, net_name, visua_path):
    assert y_true.shape == y_pred.shape, "The shape of true labels is {} {}, "
    "while the prediction is {} {}".format(y_true.shape[0], y_true[1],
                                           y_pred[0], y_pred[1])

    (n_samples, n_classes) = y_true.shape
    thresholds = np.linspace(0, 1, 1000)

    f1_score_list = list()
    max_f1_list = list()
    optimal_thresholds = list()
    for i in range(n_classes):
        max_f1 = -np.inf
        optimal_threshold = 0
        f1_list = list()
        for threshold in thresholds:
            f1 = f1_score(y_true[:, i], y_pred[:, i] > threshold)
            f1_list.append(f1)
            if f1 > max_f1:
                max_f1 = f1
                optimal_threshold = threshold

        f1_score_list.append(f1_list)
        max_f1_list.append(max_f1)
        optimal_thresholds.append(optimal_threshold)

    visua_f1_score(visua_path, net_name, f1_score_list, thresholds)

    return max_f1_list, np.array(optimal_thresholds)


def visua_f1_score(visua_path, net_name, f1, thresholds):
    fig, ax = plt.subplots(7, 4, figsize=(10, 12))
    for i in range(7):
        for j in range(4):
            if i * 4 + j < 28:
                ax[i][j].fill_between(thresholds, f1[i * 4 + j], alpha=0.5, color='b')
                ax[i][j].set_xlabel("Threshold")
                ax[i][j].set_ylabel('F1 score')
                ax[i][j].set_xlim([0.0, 1.0])
                ax[i][j].set_ylim([0.0, 1.05])
                ax[i][j].set_title("Class: {}".format(i * 4 + j))
            else:
                break

    fig.tight_layout()
    filename = os.path.join(visua_path, "{}_f1.pdf".format(net_name))
    fig.savefig(filename)


def visua_precision_recall_curve(visua_path, net_name, y_true, y_pred):
    fig, ax = plt.subplots(7, 4, figsize=(10, 12))
    for i in range(7):
        for j in range(4):
            if i * 4 + j < 28:
                precision, recall, _ = precision_recall_curve(y_true[:, i * 4 + j],
                                                              y_pred[:, i * 4 + j])
                ax[i][j].fill_between(recall, precision, alpha=0.5, color='b')
                ax[i][j].set_xlabel("Recall")
                ax[i][j].set_ylabel('Precision')
                ax[i][j].set_xlim([0.0, 1.0])
                ax[i][j].set_ylim([0.0, 1.05])
                ax[i][j].set_title("Class: {}".format(i * 4 + j))
            else:
                break

    fig.tight_layout()
    filename = os.path.join(visua_path, "{}_aupr.pdf".format(net_name))
    fig.savefig(filename)


def visua_prob_distribution(visua_path, net_name, training_prob, test_prob):
    fig, ax = plt.subplots(7, 4, figsize=(10, 12))
    for i in range(7):
        for j in range(4):
            if i * 4 + j < 28:
                sns.kdeplot(data=training_prob[:, i * 4 + j],
                            label="Validation",
                            ax=ax[i][j],
                            shade=True,
                            color="r")
                sns.kdeplot(data=test_prob[:, i * 4 + j],
                            label="Test",
                            ax=ax[i][j],
                            shade=True,
                            color="b")
                ax[i][j].set_title("Class: {}".format(i * 4 + j))
            else:
                break

    fig.tight_layout()
    filename = os.path.join(visua_path, "{}_distribution.pdf".format(net_name))
    fig.savefig(filename)


def evaluate_validation(args):
    print("load prediction of validation data...", file=sys.stderr)

    training_predicted_path = get_training_predict_path(args.net_name)
    filename = os.path.join(training_predicted_path, "{}.npz".format(args.net_name))
    assert os.path.exists(filename), "the prediction {} does not exist".format(filename)

    valid_label = np.load(filename)['label']
    valid_pred = np.load(filename)['pred']

    visua_path = os.path.join(VISUALIZATION_PATH, args.net_name)
    if not os.path.exists(visua_path):
        os.mkdir(visua_path)

    # visualize the aupr
    visua_precision_recall_curve(visua_path, args.net_name, valid_label, valid_pred)

    # get the optimal threshold from validation data
    max_f1_list, optimal_thresholds = search_threshold(y_true=valid_label, y_pred=valid_pred,
                                                       net_name=args.net_name, visua_path=visua_path)

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.bar(x=range(N_LABELS), height=max_f1_list)
    fig.tight_layout()
    filename = os.path.join(visua_path, "{}_max_f1.pdf".format(args.net_name))
    fig.savefig(filename)

    macro_f1 = np.mean(max_f1_list)

    # convert the predicted probabilities into labels for training data
    valid_predicted_labels = list()
    for i in range(valid_pred.shape[0]):
        label_predict = np.arange(N_LABELS)[np.greater(valid_pred[i], optimal_thresholds)]

        if label_predict.size == 0:
            label_predict = [np.argmax(valid_pred[i])]

        str_predict_label = " ".join(str(label) for label in label_predict)
        valid_predicted_labels.append(str_predict_label)

    df1 = pd.read_csv(TRAINING_DATA_CSV)
    df2 = pd.read_csv(HPAV18_CSV)

    df = pd.concat([df1, df2])
    df['Predicted'] = valid_predicted_labels
    filename = os.path.join(training_predicted_path, "{}_f1_{}.csv".format(args.net_name, macro_f1))
    df.to_csv(filename, index=False)

    return macro_f1, optimal_thresholds


def get_submission(args, f1=None, threshold=0.1):
    print("load prediction of test data...", file=sys.stderr)
    test_predict_path = get_test_predict_path(args.net_name)
    filename = os.path.join(test_predict_path, "{}.npz".format(args.net_name))
    assert os.path.exists(filename), "the prediction {} does not exist".format(filename)

    test_pred = np.load(filename)['pred']

    submission_path = get_submission_path(net_name=args.net_name)

    # convert the predicted probabilities into labels for training data
    output_test_labels = list()
    for i in range(test_pred.shape[0]):
        label_predict = np.arange(N_LABELS)[np.greater(test_pred[i], threshold)]

        if label_predict.size == 0:
            label_predict = [np.argmax(test_pred[i])]

        str_predict_label = " ".join(str(label) for label in label_predict)
        output_test_labels.append(str_predict_label)

    df = pd.read_csv(SAMPLE_SUBMISSION)
    df['Predicted'] = output_test_labels
    filename = os.path.join(submission_path, "{}_val_f1_{}.csv".format(args.net_name, f1))
    df.to_csv(filename, index=False)


if __name__ == '__main__':
    arguments = parse_args()
    f1, thres = evaluate_validation(arguments)
    print(thres)
    get_submission(arguments, f1, thres)
