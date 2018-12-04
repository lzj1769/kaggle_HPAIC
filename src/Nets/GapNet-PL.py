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

BATCH_SIZE = 8
INPUT_SHAPE = (2048, 2048, 4)
MAX_QUEUE_SIZE = 10


def build_model(num_classes=None,
                input_shape=INPUT_SHAPE):
    img_input = layers.Input(shape=input_shape, name='data')

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.Conv2D(filters=64,
                      kernel_size=(3, 3),
                      strides=(2, 2),
                      kernel_initializer='he_normal',
                      padding='valid',
                      name='conv1')(img_input)

    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    gap1 = layers.GlobalAveragePooling2D()(x)

    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2))(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv2')(x)

    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1))(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv3')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    gap2 = layers.GlobalAveragePooling2D()(x)

    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1))(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv4')(x)

    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1))(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv5')(x)

    gap3 = layers.GlobalAveragePooling2D()(x)

    x = layers.Concatenate()([gap1, gap2, gap3])

    x = layers.Dense(1024, activation='relu', name='fc1024_1')(x)
    x = layers.BatchNormalization(name="batch_1")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation='relu', name='fc1024_2')(x)
    x = layers.BatchNormalization(name="batch_2")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation='sigmoid', name='fc28')(x)

    # this is the model we will train
    model = Model(inputs=img_input, outputs=x, name='GapNet-PL')

    return model


model = build_model(num_classes=28)

model.summary()
