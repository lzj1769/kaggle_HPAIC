from __future__ import print_function, division
import numpy as np
import pandas as pd

from configure import *

from visualization import visua_threshold_f1
from visualization import visua_f1_classes
from visualization import visua_prob_distribution
from visualization import visua_cnn


def test_visua_threshold_f1():
    f1_score = list()
    optimal_threshold = list()
    for i in range(N_LABELS):
        f1_score.append(np.random.rand(100))
        optimal_threshold.append(np.random.rand())

    visua_threshold_f1(f1_score, optimal_threshold, "test")


def test_visua_f1_classes():
    f1_score = np.random.rand(N_LABELS)
    visua_f1_classes(f1_score, "test")


def test_visua_prob_distribution():
    visua_path = VISUALIZATION_PATH
    exp_config = "test"
    training_prob = np.load("/home/rs619065/kaggle_HPAIC/training/ResNet50/ResNet50_PreTrained_KFold_0.npz")['pred']
    test_prob = np.load("/home/rs619065/kaggle_HPAIC/test/ResNet50/ResNet50_PreTrained_KFold_0.npz")['pred']

    visua_prob_distribution(visua_path, exp_config, training_prob, test_prob)


def test_visua_cnn():
    from keras.models import load_model
    from keras.utils.io_utils import h5dict
    from utils import load_data, get_custom_objects
    df = pd.read_csv(SAMPLE_SUBMISSION)
    custom_objects = get_custom_objects('ResNet50')
    weights_filename = "GapNet-PL_KFold_2.h5"
    f = h5dict(weights_filename, 'r')
    print(f.get('training_config'))
    f.close()
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = load_model(weights_filename, custom_objects=custom_objects)
    model.summary()
    image = load_data(TEST_DATA_1024)
    for i in range(100, 150, 1):
       visua_cnn(model=model, image=image[i], id=df['Id'][i])


test_visua_cnn()
