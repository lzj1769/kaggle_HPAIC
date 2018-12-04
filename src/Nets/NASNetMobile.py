from keras_applications.nasnet import NASNetMobile

import keras
from keras import Model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import MaxPooling2D, GlobalAveragePooling2D

WEIGHTS_PATH = '/home/rs619065/.keras/models/nasnet_mobile_no_top.h5'
BATCH_SIZE = 12
INPUT_SHAPE = (1024, 1024, 3)
MAX_QUEUE_SIZE = 20


def build_model(num_classes):
    # create the base DenseNet121-trained model
    base_model = NASNetMobile(weights=WEIGHTS_PATH,
                              include_top=False,
                              input_shape=INPUT_SHAPE,
                              backend=keras.backend,
                              layers=keras.layers,
                              models=keras.models,
                              utils=keras.utils)

    # add a global spatial average pooling layer
    x = base_model.output
    x = MaxPooling2D((3, 3), strides=(2, 2), name='max_pool')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu', name='fc1024_1')(x)
    x = BatchNormalization(name="batch_1")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', name='fc1024_2')(x)
    x = BatchNormalization(name="batch_2")(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='sigmoid', name='fc28')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=x, name='NASNetMobile')

    return model
