from keras_applications.densenet import DenseNet201
from keras_applications.densenet import preprocess_input

import keras
from keras import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import SGD

from callback import build_callbacks
from loss import focal_loss

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


def build_model(input_shape, num_classes, weights='imagenet'):
    # create the base pre-trained model
    base_model = DenseNet201(weights=weights, include_top=False, input_shape=input_shape,
                             backend=keras.backend, layers=keras.layers, models=keras.models,
                             utils=keras.utils)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='sigmoid', name='fc28')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=x, name='pre_trained_densenet201')

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=sgd, loss=focal_loss, metrics=['accuracy'])

    model.summary()

    return model
