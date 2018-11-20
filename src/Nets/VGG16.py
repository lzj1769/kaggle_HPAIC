import sys
from keras_applications.vgg16 import VGG16

import keras
from keras import Model
from keras.layers import Dense, Dropout, BatchNormalization

sys.setrecursionlimit(3000)

WEIGHTS_PATH = '/home/rs619065/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
TRAINING_DATA = "/hpcwork/izkf/projects/SingleCellOpenChromatin/HPAIC/data/train_2048.npy"
TEST_DATA = "/hpcwork/izkf/projects/SingleCellOpenChromatin/HPAIC/test_2048.npy"
BATCH_SIZE = 4
INPUT_SHAPE = (2048, 2048, 3)


def build_model(num_classes):
    # create the base pre-trained model
    base_model = VGG16(weights=WEIGHTS_PATH,
                       include_top=False,
                       input_shape=INPUT_SHAPE,
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
