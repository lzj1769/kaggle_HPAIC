from __future__ import print_function
from __future__ import division

import os
import sys

import pandas as pd
import numpy as np
import argparse

from sklearn.metrics import f1_score

from configure import *
from utils import get_submission_path, get_training_predict_path, get_test_predict_path
from visualization import visua_prob_distribution

parser = argparse.ArgumentParser()
parser.add_argument("--net_name", help='name of convolutional neural network', default=None)
parser.add_argument("--verbose", type=int, default=2, help="Verbosity mode. DEFAULT: 2")

args = parser.parse_args()

print("load prediction of validation data...", file=sys.stderr)

# training_predicted_path = get_training_predict_path(args.net_name)
# filename = os.path.join(training_predicted_path, "{}.npz".format(args.net_name))
# assert os.path.exists(filename), "the prediction {} does not exist".format(filename)
#
# training_pred = np.load(filename)['pred']
# training_label = np.load(filename)['label']

# thresholds = np.linspace(0, 1, 1000)
# max_f1 = 0.0
# optimal_threshold = 0.0
# for threshold in thresholds:
#     f1 = f1_score(y_true=training_label, y_pred=training_pred > threshold, average="macro").round(3)
#
#     if f1 > max_f1:
#         optimal_threshold = threshold
#         max_f1 = f1

# f1 score for validation data set
# f1 = f1_score(y_true=training_label, y_pred=training_pred > THRESHOLD, average="macro").round(3)
#
# # convert the predicted probabilities into labels for training data
# training_predicted_labels = list()
# for i in range(training_pred.shape[0]):
#     label_predict = np.arange(N_LABELS)[np.greater(training_pred[i], THRESHOLD)]
#
#     str_predict_label = " ".join(str(label) for label in label_predict)
#     training_predicted_labels.append(str_predict_label)
#
# df = pd.read_csv(TRAINING_DATA_CSV)
# df['Predicted'] = training_predicted_labels
# filename = os.path.join(training_predicted_path, "{}_f1_{}.csv".format(args.net_name, f1))
# df.to_csv(filename, index=False)
#
# print("load prediction of test data...", file=sys.stderr)
# df = pd.read_csv(SAMPLE_SUBMISSION)

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

# visua_prob_distribution(VISUALIZATION_PATH, args.net_name, training_pred, test_pred)
