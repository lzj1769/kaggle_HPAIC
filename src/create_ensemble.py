from __future__ import print_function

import os
import numpy as np
from configure import *
from utils import get_target
from utils import get_training_predict_path
from utils import get_test_predict_path

NET_NAMES = ['ResNet50', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'InceptionV3']

ensemble_training_path = os.path.join(TRAINING_OUTPUT_PATH, "Ensemble")

if not os.path.exists(ensemble_training_path):
    os.mkdir(ensemble_training_path)

ensemble_test_path = os.path.join(TEST_OUTPUT_PATH, "Ensemble")

if not os.path.exists(ensemble_test_path):
    os.mkdir(ensemble_test_path)

x_train = None
x_test = None
for net_name in NET_NAMES:
    training_predicted_path = get_training_predict_path(net_name)
    filename = os.path.join(training_predicted_path, "{}.npz".format(net_name))
    assert os.path.exists(filename), "the prediction {} does not exist".format(filename)

    pred = np.load(filename)['pred']

    if x_train is None:
        x_train = pred
    else:
        x_train = np.append(x_train, pred, axis=1)

    test_predicted_path = get_test_predict_path(net_name)
    filename = os.path.join(test_predicted_path, "{}.npz".format(net_name))
    assert os.path.exists(filename), "the prediction {} does not exist".format(filename)

    pred = np.load(filename)['pred']

    if x_test is None:
        x_test = pred
    else:
        x_test = np.append(x_test, pred, axis=1)

filename = os.path.join(ensemble_training_path, "Ensemble.npz")
np.savez(file=filename, pred=x_train, label=get_target())

filename = os.path.join(ensemble_test_path, "Ensemble.npz")
np.savez(file=filename, pred=x_test)
