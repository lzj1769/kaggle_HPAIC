"""GapNet-PL model for Keras.

# Reference:

- [Human-level Protein Localization with Convolutional Neural Networks](https://openreview.net/forum?id=ryl5khRcKm)

Implementation is based on Keras 2.0
"""
from keras.models import Model
from keras import backend as K
from keras import layers

import sys

sys.setrecursionlimit(3000)

BATCH_SIZE = 16
INPUT_SHAPE = (1024, 1024, 4)
MAX_QUEUE_SIZE = 32
LEARNING_RATE = 1e-04


def build_model(num_classes=None,
                input_shape=INPUT_SHAPE):
    img_input = layers.Input(shape=input_shape, name='data')

    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), name='conv1')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    gap1 = layers.GlobalAveragePooling2D()(x)

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    gap2 = layers.GlobalAveragePooling2D()(x)

    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    gap3 = layers.GlobalAveragePooling2D()(x)

    x = layers.Concatenate()([gap1, gap2, gap3])

    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu', name='fc2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(num_classes, activation='sigmoid', name='fc28')(x)

    # this is the model we will train
    model = Model(inputs=img_input, outputs=x, name='GapNet-PL')

    return model

