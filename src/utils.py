from __future__ import print_function, division
import os
import sys
import numpy as np

from sklearn.metrics import f1_score


from configure import *


def load_data(dataset=None):
    if dataset == "test":
        img = np.load(TEST_DATA)['img']

        return img

    elif dataset == "train":
        img = np.load(TRAINING_DATA)['img']
        labels = np.load(TRAINING_DATA)['label']

        return img, labels

    else:
        print("the data set doesn't exist...", file=sys.stderr)
        exit(1)


def get_input_shape(net_name, pre_trained=True):
    input_shape = None

    if pre_trained:
        if net_name in ['DenseNet121', 'DenseNet119', 'DenseNet201', 'MobileNet', 'MobileNetV2',
                        'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']:
            input_shape = (224, 224, 3)

        elif net_name in ['InceptionResNetV2', 'InceptionV3', 'Xception']:
            input_shape = (299, 299, 3)

        elif net_name in ['NASNetLarge', 'NASNetMobile']:
            input_shape = (331, 331, 3)

        else:
            print("Network {} doesn't exist".format(net_name))

    else:
        input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)

    return input_shape


def get_batch_size(net_name, pre_trained=True):
    if pre_trained:
        if net_name in ['NASNetLarge']:
            batch_size = 8
        else:
            batch_size = 16

    else:
        batch_size = 8

    return batch_size


def generate_exp_config(net_name, pre_trained, loss, k_fold=None):
    exp_config = net_name
    if pre_trained:
        exp_config += "_PreTrained_"
    else:
        exp_config += "_FromScratch_"

    exp_config += LOSS[loss]

    if k_fold is not None:
        return "{}_KFold_{}".format(exp_config, k_fold)
    else:
        return exp_config


def get_logs_path(net_name):
    return os.path.join(MODEL_LOG_PATH, net_name)


def get_weights_path(net_name):
    return os.path.join(MODEL_WEIGHTS_PATH, net_name)


def get_acc_loss_path(net_name):
    return os.path.join(MODEL_ACC_LOSS_PATH, net_name)


def get_training_predict_path(net_name):
    return os.path.join(TRAINING_OUTPUT_PATH, net_name)


def get_test_predict_path(net_name):
    return os.path.join(TEST_OUTPUT_PATH, net_name)


def get_submission_path(net_name):
    return os.path.join(SUBMISSION_PATH, net_name)


def optimal_threshold(y_true, y_prab):
    assert y_true.shape == y_prab.shape, print(
        "The shape of true labels is {} {}, while the prediction is {} {}".format(y_true.shape[0], y_true[1],
                                                                                  y_prab[0], y_prab[1]))
    (n_samples, n_classes) = y_true.shape
    thresholds = np.linspace(0, 1, 1000)

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
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        f1_scores.append(f1)

    return f1_scores


def calculate_threshold(y_pred, fraction):
    threshod = []

    for i in range(N_LABELS):
        prab = y_pred[:, i]
        frac = fraction[i]

        threshod.append(np.quantile(prab, 1 - frac))

    return np.array(threshod)
