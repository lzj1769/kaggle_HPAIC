import sys
from keras_applications.vgg19 import VGG19

import keras
from keras import Model
from keras.layers import Dense, Dropout, BatchNormalization

sys.setrecursionlimit(3000)

WEIGHTS_PATH = '/home/rs619065/.keras/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
TRAINING_DATA = "/hpcwork/izkf/projects/SingleCellOpenChromatin/HPAIC/data/train_1024.npy"
TEST_DATA = "/hpcwork/izkf/projects/SingleCellOpenChromatin/HPAIC/data/test_1024.npy"
BATCH_SIZE = 8
INPUT_SHAPE = (1024, 1024, 3)


def build_model(num_classes):
    # create the base pre-trained model
    base_model = VGG19(weights=WEIGHTS_PATH,
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
    model = Model(inputs=base_model.input, outputs=x, name='VGG19')

    return model