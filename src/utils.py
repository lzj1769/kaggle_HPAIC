from __future__ import print_function, division
import sys
import numpy as np
import pandas as pd
import time
import platform

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

import tensorflow as tf

from pynvml import (nvmlInit,
                    nvmlDeviceGetCount,
                    nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetUtilizationRates,
                    nvmlDeviceGetName)

from configure import *


def load_data(dataset=None):
    mlb = MultiLabelBinarizer(classes=range(N_LABELS))

    if dataset == "test":
        img = np.load(TEST_DATA)['img']

        return img

    elif dataset == "validation":
        img = np.load(VALIDATION_DATA)['img']
        df = pd.read_csv(VALIDATION_DATA_CSV)

        labels = list()
        for target in df['Target']:
            label = map(int, target.split(" "))
            labels.append(label)

        return img, mlb.fit_transform(labels)

    elif dataset == "train":
        img = np.load(TRAINING_DATA)['img']
        df = pd.read_csv(TRAINING_DATA_CSV)

        labels = list()
        for target in df['Target']:
            label = map(int, target.split(" "))
            labels.append(label)

        return img, mlb.fit_transform(labels)

    else:
        print("the data set doesn't exist...", file=sys.stderr)
        exit(1)


def hms(seconds):
    seconds = np.floor(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return "%02d:%02d:%02d" % (hours, minutes, seconds)


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def hostname():
    return platform.node()


def generate_expid(model_name):
    return "%s-%s" % (model_name, timestamp())


def gpu_info():
    "Returns a tuple of (GPU ID, GPU Description, GPU % Utilization)"
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    info = []
    for i in range(0, deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        util = nvmlDeviceGetUtilizationRates(handle)
        desc = nvmlDeviceGetName(handle)
        info.append((i, desc, util.gpu))
    return info


def optimal_threshold(y_true, y_prab):
    assert y_true.shape == y_prab.shape, print(
        "The shape of true labels is {} {}, while the prediction is {} {}".format(y_true.shape[0], y_true[1],
                                                                                  y_prab[0], y_prab[1]))
    (n_samples, n_classes) = y_true.shape
    thresholds = np.linspace(0, 1, 100)

    f1_scores_list = list()
    optimal_thresholds = list()
    optimal_f1_score = list()
    for i in range(n_classes):
        f1_scores = f1_scores_threshold(y_true[:, i], y_prab[:, i], thresholds)
        f1_scores_list.append(f1_scores)
        idx = np.argmax(f1_scores)
        optimal_thresholds.append(thresholds[idx])
        optimal_f1_score.append(f1_scores[idx])

    return f1_scores_list, np.array(optimal_thresholds), optimal_f1_score


def f1_scores_threshold(y_true, y_prab, thresholds):
    f1_scores = []
    for threshold in thresholds:
        y_pred = y_prab > threshold
        f1 = f1_score(y_true=y_true.tolist(), y_pred=y_pred.tolist())
        f1_scores.append(f1)

    return f1_scores


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)