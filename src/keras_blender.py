from __future__ import print_function, division

import os
import sys

import pandas as pd
import numpy as np
import argparse
from keras import layers
from keras import Model

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


def build_model(input_shape, num_classes):
    input_data = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=8, kernel_size=(3, 1), strides=(1, 1))(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=16, kernel_size=(3, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu', name='fc1')(x)
    x = layers.Dense(num_classes, activation='sigmoid', name='fc28')(x)

    model = Model(inputs=input_data, outputs=x, name='Keras_blender')

    return model
