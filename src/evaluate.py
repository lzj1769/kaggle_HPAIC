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

from sklearn.metrics import f1_score

from configure import *
from utils import get_submission_path, get_training_predict_path, get_test_predict_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_name", help='name of convolutional neural network', default=None)
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity mode. DEFAULT: 2")
    return parser.parse_args()


def search_threshold(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "The shape of true labels is {} {}, "
    "while the prediction is {} {}".format(y_true.shape[0], y_true[1],
                                           y_pred[0], y_pred[1])

    (n_samples, n_classes) = y_true.shape
    thresholds = np.linspace(0, 1, 100)

    max_f1_list = list()
    optimal_thresholds = list()
    for i in range(n_classes):
        max_f1 = -np.inf
        optimal_threshold = 0
        for threshold in thresholds:
            f1 = f1_score(y_true[:, i], y_pred[:, i] > threshold)
            if f1 > max_f1:
                max_f1 = f1
                optimal_threshold = threshold

        max_f1_list.append(max_f1)
        optimal_thresholds.append(optimal_threshold)

    return max_f1_list, np.array(optimal_thresholds)


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
    filename = os.path.join(visua_path, "{}.pdf".format(net_name))
    fig.savefig(filename)


def evaluate_validation(args):
    print("load prediction of validation data...", file=sys.stderr)

    training_predicted_path = get_training_predict_path(args.net_name)
    filename = os.path.join(training_predicted_path, "{}.npz".format(args.net_name))
    assert os.path.exists(filename), "the prediction {} does not exist".format(filename)

    valid_label = np.load(filename)['label']
    valid_pred = np.load(filename)['pred']

    # get the optimal threshold from validation data
    max_f1_list, optimal_thresholds = search_threshold(y_true=valid_label, y_pred=valid_pred)

    macro_f1 = np.mean(max_f1_list)

    # convert the predicted probabilities into labels for training data
    valid_predicted_labels = list()
    for i in range(valid_pred.shape[0]):
        label_predict = np.arange(N_LABELS)[np.greater(valid_pred[i], optimal_thresholds)]
        str_predict_label = " ".join(str(label) for label in label_predict)
        valid_predicted_labels.append(str_predict_label)

    df = pd.read_csv(TRAINING_DATA_CSV)
    df['Predicted'] = valid_predicted_labels
    filename = os.path.join(training_predicted_path, "{}_f1_{}.csv".format(args.net_name, macro_f1))
    df.to_csv(filename, index=False)

    return macro_f1, optimal_thresholds


def get_submission(args):
    training_predicted_path = get_training_predict_path(args.net_name)
    filename = os.path.join(training_predicted_path, "{}.npz".format(args.net_name))
    assert os.path.exists(filename), "the prediction {} does not exist".format(filename)

    valid_pred = np.load(filename)['pred']

    print("load prediction of test data...", file=sys.stderr)
    test_predict_path = get_test_predict_path(args.net_name)
    filename = os.path.join(test_predict_path, "{}.npz".format(args.net_name))
    assert os.path.exists(filename), "the prediction {} does not exist".format(filename)

    test_pred = np.load(filename)['pred']

    submission_path = get_submission_path(net_name=args.net_name)

    # convert the predicted probabilities into labels for training data
    output_test_labels = list()
    for i in range(test_pred.shape[0]):
        label_predict = np.arange(N_LABELS)[np.greater(test_pred[i], THRESHOLD)]
        str_predict_label = " ".join(str(label) for label in label_predict)
        output_test_labels.append(str_predict_label)

    df = pd.read_csv(SAMPLE_SUBMISSION)
    df['Predicted'] = output_test_labels
    filename = os.path.join(submission_path, "{}.csv".format(args.net_name))
    df.to_csv(filename, index=False)

    visua_prob_distribution(VISUALIZATION_PATH, args.net_name, valid_pred, test_pred)


if __name__ == '__main__':
    arguments = parse_args()
    #f1, thres = evaluate_validation(arguments)
    #print(thres)
    get_submission(arguments)
