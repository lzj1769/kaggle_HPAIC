from __future__ import print_function
from __future__ import division

import os
import sys

import pandas as pd
import numpy as np
import argparse

from configure import *
from utils import get_training_predict_path, get_test_predict_path
from utils import generate_exp_config, calculate_threshold
from utils import optimal_threshold
from visualization import visua_prob_distribution


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_name", help='name of convolutional neural network', default=None)
    parser.add_argument("--pre_trained", type=int, default=1,
                        help="whether use the pre-trained weights or not, set 0 will train the network from "
                             "scratch and 1 will use the weights from imagenet. DEFAULT: 1")
    parser.add_argument("--include_fc", type=int, default=0,
                        help="whether include the full connect layers for trained neural network. DEFAULT 0")
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity mode. DEFAULT: 2")
    return parser.parse_args()


def main():
    args = parse_args()

    print("load the model configuration...", file=sys.stderr)
    print("=======================================================", file=sys.stderr)

    train_predict_path = get_training_predict_path(args.net_name)

    df_train = pd.read_csv(TRAINING_DATA_CSV)
    train_pred = np.zeros((df_train.shape[0], N_LABELS))
    train_true = np.zeros((df_train.shape[0], N_LABELS))

    for k_fold in range(N_SPLIT):
        exp_config = generate_exp_config(args.net_name, args.pre_trained, args.include_fc, k_fold)
        filename = os.path.join(train_predict_path, "{}.npz".format(exp_config))
        assert os.path.exists(filename), print("the prediction {} does not exist".format(filename))
        pred = np.load(filename)['pred']
        label = np.load(filename)['label']

        split_filename = os.path.join(DATA_DIR, "KFold_{}.npz".format(k_fold))
        split = np.load(split_filename)

        test_indexes = split['test_indexes']

        for i, idx in enumerate(test_indexes):
            train_pred[idx] = pred[i]
            train_true[idx] = label[i]

    print("calculating the threshold according to fraction of each label...", file=sys.stderr)

    f1_scores_list, optimal_thresholds, optimal_f1_score = optimal_threshold(y_true=train_true, y_prab=train_pred)
    mean_f1 = np.mean(optimal_f1_score)

    print(mean_f1)

    fraction = []
    for i in range(train_pred.shape[1]):
        thres = optimal_thresholds[i]
        pred = train_pred[:, i]
        fraction.append(np.sum(pred > thres) / len(pred))

    print(fraction)

    df = pd.read_csv(SAMPLE_SUBMISSION)

    test_predcit_path = get_test_predict_path(args.net_name)
    test_pred = np.zeros((df.shape[0], N_LABELS))

    for k_fold in range(N_SPLIT):
        exp_config = generate_exp_config(args.net_name, args.pre_trained, args.include_fc, k_fold)
        filename = os.path.join(test_predcit_path, "{}.npz".format(exp_config))
        assert os.path.exists(filename), print("the prediction {} does not exist".format(filename))
        test_pred += np.load(filename)['pred']

    test_pred /= N_SPLIT

    exp_config = generate_exp_config(args.net_name, args.pre_trained, args.include_fc)
    visua_prob_distribution(VISUALIZATION_PATH, exp_config, train_pred, test_pred)

    test_thres = calculate_threshold(test_pred, fraction=fraction)

    output_test_labels = list()

    # convert the predicted probabilities into labels for training data
    for i in range(test_pred.shape[0]):
        label_predict = np.arange(N_LABELS)[np.greater(test_pred[i], test_thres)]
        if label_predict.size == 0:
            label_predict = [np.argmax(test_pred[i])]

        str_predict_label = " ".join(str(label) for label in label_predict)
        output_test_labels.append(str_predict_label)

    df['Predicted'] = output_test_labels

    exp_config = generate_exp_config(args.net_name, args.pre_trained, args.include_fc)
    submission_filename = "{}.csv".format(exp_config)
    filename = os.path.join(SUBMISSION_PATH, submission_filename)
    df.to_csv(filename, index=False)


if __name__ == '__main__':
    main()
