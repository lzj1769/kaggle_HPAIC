from __future__ import print_function

import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation
from keras.optimizers import SGD
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from callback import EarlyStopping

epochs = 300
batch_size = 16

augment = True
use_multiprocessing = True

INPUT_SHAPE = (512, 512, 4)

augment_parameters = {'rotation_range': 180,
                      'width_shift_range': 0.2,
                      'height_shift_range': 0.2,
                      'brightness_range': None,
                      'shear_range': 0.2,
                      'zoom_range': 0.4,
                      'channel_shift_range': 10,
                      'fill_mode': 'nearest',
                      'cval': 0.,
                      'horizontal_flip': True,
                      'vertical_flip': True}


def build_model(input_shape, num_classes, weights=None):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

    model.compile(loss=binary_crossentropy, optimizer=sgd,
                  metrics=['accuracy'])

    model.summary()

    if weights is not None:
        model.load_weights(weights)

    return model


def build_callbacks(model_path, net_name):
    fp = os.path.join(model_path, "{}.h5".format(net_name))
    check_pointer = ModelCheckpoint(filepath=fp,
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True)

    early_stopper = EarlyStopping(monitor='val_loss',
                                  patience=20,
                                  seconds=3600 * 7,
                                  verbose=1,
                                  restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=5,
                                  min_lr=1e-08,
                                  min_delta=0.,
                                  verbose=1)

    callbacks = [check_pointer, early_stopper, reduce_lr]

    return callbacks

# model = build_model((IMAGE_HEIGHT, IMAGE_WIDTH, 4), N_LABELS)
