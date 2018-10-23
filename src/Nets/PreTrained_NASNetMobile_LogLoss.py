import os

from keras_applications.nasnet import NASNetMobile
from keras_applications.nasnet import preprocess_input

import keras
from keras import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import SGD
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from callback import EarlyStopping

epochs = 300
batch_size = 32

augment = True
use_multiprocessing = True
INPUT_SHAPE = (224, 224, 3)
preprocess_input = preprocess_input

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


def build_model(input_shape, num_classes, weights='imagenet'):
    # create the base pre-trained model
    base_model = NASNetMobile(weights=weights, include_top=False, input_shape=input_shape,
                             backend=keras.backend, layers=keras.layers, models=keras.models,
                             utils=keras.utils)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='sigmoid', name='fc28')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=x, name='pre_trained_NASNetMobile')

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=sgd, loss=binary_crossentropy, metrics=['accuracy'])

    model.summary()

    return model
