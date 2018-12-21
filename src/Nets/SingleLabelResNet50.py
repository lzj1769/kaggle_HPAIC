import sys
from keras.applications import ResNet50

import keras
from keras import Model
from keras.layers import Dense, Dropout, BatchNormalization

sys.setrecursionlimit(3000)

WEIGHTS_PATH = '/home/rs619065/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
BATCH_SIZE = 32
INPUT_SHAPE = (512, 512, 3)
MAX_QUEUE_SIZE = 32
LEARNING_RATE = 1e-04


def build_model(num_classes):
    base_model = ResNet50(weights=WEIGHTS_PATH,
                          include_top=False,
                          input_shape=INPUT_SHAPE,
                          backend=keras.backend,
                          layers=keras.layers,
                          models=keras.models,
                          utils=keras.utils,
                          pooling='avg')

    x = base_model.output

    x = Dense(512, activation='relu', name='fc1')(x)
    x = BatchNormalization(name="batch_1")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = BatchNormalization(name="batch_2")(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='sigmoid', name='fc28')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=x, name='ResNet50')

    return model