from keras_applications.mobilenet import MobileNet
from keras_applications.mobilenet import preprocess_input

import keras
from keras import Model
from keras.layers import GlobalAveragePooling2D, Reshape, Dropout
from keras.layers import Conv2D, Activation
from keras.optimizers import SGD
from loss import focal_loss

from callback import build_callbacks

epochs = 300
batch_size = 32

augment = True
use_multiprocessing = True
INPUT_SHAPE = (224, 224, 3)
preprocess_input = preprocess_input
build_callbacks = build_callbacks

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


def build_model(input_shape, num_classes, weights='imagenet', alpha=1.0, dropout=1e-3):
    # create the base pre-trained model
    base_model = MobileNet(weights=weights, include_top=False, input_shape=input_shape,
                           backend=keras.backend, layers=keras.layers, models=keras.models,
                           utils=keras.utils)

    # add a global spatial average pooling layer
    shape = (1, 1, int(1024 * alpha))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Reshape(shape, name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(num_classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('sigmoid', name='act_sigmoid')(x)
    x = Reshape((num_classes,), name='reshape_2')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=x, name='pre_trained_densenet121')

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=sgd, loss=focal_loss, metrics=['accuracy'])

    model.summary()

    return model
