from keras_applications.mobilenet_v2 import MobileNetV2
from keras_applications.mobilenet_v2 import preprocess_input

import keras
from keras import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.optimizers import SGD
from keras.losses import binary_crossentropy

preprocess_input = preprocess_input


def build_model(input_shape, num_classes, weights='imagenet', include_fc=False):
    # create the base pre-trained model

    base_model = MobileNetV2(weights=weights, include_top=False, input_shape=input_shape,
                             backend=keras.backend, layers=keras.layers, models=keras.models,
                             utils=keras.utils)

    if include_fc:
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        x = Dense(1024, activation='relu', name='fc2014_1')(x)
        x = Dropout(0.2)(x)
        x = Dense(1024, activation='relu', name='fc2014_2')(x)
        x = Dropout(0.2)(x)
        x = Dense(num_classes, activation='sigmoid', name='fc28')(x)
    else:
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(num_classes, activation='sigmoid', name='fc28')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=x, name='pre_trained_resnet50')

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True, decay=1e-06)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=sgd, loss=binary_crossentropy, metrics=['accuracy'])

    return model
