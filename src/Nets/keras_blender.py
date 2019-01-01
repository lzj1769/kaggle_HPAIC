from __future__ import print_function, division

import os
import sys

import pandas as pd
import numpy as np
import argparse
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.layers.advanced_activations import PReLU
from keras.losses import binary_crossentropy
from sklearn.model_selection import KFold

from utils import get_training_predict_path
from utils import get_test_predict_path
from utils import get_target
from configure import *

NET_NAMES = ['ResNet50', 'DenseNet121', 'DenseNet169', 'DenseNet201']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_models", type=int, default=100, help='Number of CNNs. DEFAULT: 100')
    parser.add_argument("-v", "--verbose", type=int, default=2, help="Verbosity mode. DEFAULT: 2")
    return parser.parse_args()


def blender(input_dim, n_classes, params):
    level1_size = 300
    if 'level1_size' in params:
        level1_size = params['level1_size']
    level2_size = 300
    if 'level2_size' in params:
        level2_size = params['level2_size']
    level3_size = 250
    if 'level3_size' in params:
        level3_size = params['level3_size']

    dropout_val_1 = 0.1
    if 'dropout_val_1' in params:
        dropout_val_1 = params['dropout_val_1']
    dropout_val_2 = 0.1
    if 'dropout_val_2' in params:
        dropout_val_2 = params['dropout_val_2']
    dropout_val_3 = 0.1
    if 'dropout_val_3' in params:
        dropout_val_3 = params['dropout_val_3']

    activation_1 = 'prelu'
    if 'activation_1' in params:
        activation_1 = params['activation_1']
    activation_2 = 'prelu'
    if 'activation_2' in params:
        activation_2 = params['activation_2']
    activation_3 = 'prelu'
    if 'activation_3' in params:
        activation_3 = params['activation_3']

    model = Sequential()
    model.add(Dense(units=level1_size, input_dim=input_dim))

    if activation_1 == 'prelu':
        model.add(PReLU())
    elif activation_1 == 'relu':
        model.add(Activation('relu'))
    else:
        model.add(Activation('elu'))

    model.add(Dropout(dropout_val_1))
    model.add(Dense(level2_size))

    if activation_2 == 'prelu':
        model.add(PReLU())
    elif activation_2 == 'relu':
        model.add(Activation('relu'))
    else:
        model.add(Activation('elu'))
    model.add(Dropout(dropout_val_2))

    model.add(Dense(level3_size))
    if activation_3 == 'prelu':
        model.add(PReLU())
    elif activation_3 == 'relu':
        model.add(Activation('relu'))
    else:
        model.add(Activation('elu'))
    model.add(Dropout(dropout_val_3))

    model.add(Dense(n_classes, activation='sigmoid'))

    return model


def random_keras_step(X_train, y_train, random_state, n_iter):
    params = dict()
    num_folds = np.random.randint(4, 10)
    batch_size = np.random.randint(200, 1000)

    patience = np.random.randint(50, 150)
    learning_rate = np.random.uniform(0.00001, 0.001)

    params['dropout_val_1'] = np.random.uniform(0.05, 0.5)
    params['dropout_val_2'] = np.random.uniform(0.1, 0.5)
    params['dropout_val_3'] = np.random.uniform(0.1, 0.5)
    params['level1_size'] = np.random.randint(400, 700)
    params['level2_size'] = np.random.randint(350, 600)
    params['level3_size'] = np.random.randint(200, 500)
    params['activation_1'] = np.random.choice(['prelu', 'relu', 'elu'])
    params['activation_2'] = np.random.choice(['prelu', 'relu', 'elu'])
    params['activation_3'] = np.random.choice(['prelu', 'relu', 'elu'])

    log_str = 'Keras iter {}. FOLDS: {} LR: {}, PATIENCE: {}, BATCH: {}'.format(
        iter,
        num_folds,
        learning_rate,
        patience,
        batch_size)
    print(log_str, file=sys.stdout)
    print('CNN params: {}'.format(params), file=sys.stdout)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    num_fold = 0
    for train_index, test_index in kf.split(list(range(len(X_train[0])))):
        num_fold += 1
        print('Start fold {} from {}'.format(num_fold, num_folds))
        X_train = X_train[train_index]
        X_valid = X_train[test_index]
        y_train = y_train[train_index]
        y_valid = y_train[test_index]
        print('Shape train: ', X_train.shape, file=sys.stdout)
        print('Shape valid: ', X_valid.shape, file=sys.stdout)

        final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format("blender", num_fold)
        cache_model_path = MODELS_PATH + '{}_temp_fold_{}.h5'.format("blender", num_fold)

        model = blender(X_train.shape[1], y_train.shape[1], params)
        optimizer = Adam(lr=learning_rate)


def run_multiple_keras_blenders(args):
    for i in range(args.n_models):
        random_state = 2018 + i
        validation_arr, test_preds, log_str, cnn_param = random_keras_step(training_data=training_data,
                                                                           random_state=random_state,
                                                                           n_iter=i)
