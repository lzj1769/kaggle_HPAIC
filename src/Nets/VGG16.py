import sys
from keras_applications.vgg16 import VGG16

import keras
from keras import Model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import MaxPooling2D, GlobalAveragePooling2D

sys.setrecursionlimit(3000)

batch_size = 4
input_shape = (2048, 2048, 3)


def build_model(num_classes, weights='imagenet'):
    # create the base pre-trained model
    base_model = VGG16(weights=weights,
                       include_top=False,
                       input_shape=input_shape,
                       backend=keras.backend,
                       layers=keras.layers,
                       models=keras.models,
                       utils=keras.utils,
                       pooling='avg')

    # add a global spatial average pooling layer
    x = base_model.output
    x = Dense(1024, activation='relu', name='fc1024_1')(x)
    x = BatchNormalization(name="batch_1")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', name='fc1024_2')(x)
    x = BatchNormalization(name="batch_2")(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='sigmoid', name='fc28')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=x, name='VGG16')

    return model
