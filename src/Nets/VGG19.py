import sys
from keras_applications.vgg19 import VGG19

import keras
from keras import Model
from keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Concatenate

sys.setrecursionlimit(3000)

WEIGHTS_PATH = '/home/rs619065/.keras/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
BATCH_SIZE = 8
INPUT_SHAPE = (1024, 1024, 3)
MAX_QUEUE_SIZE = 32
LEARNING_RATE = 5e-04


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

    gap1 = GlobalAveragePooling2D()(base_model.get_layer('block1_conv2').output)
    gap2 = GlobalAveragePooling2D()(base_model.get_layer('block2_conv2').output)
    gap3 = GlobalAveragePooling2D()(base_model.get_layer('block3_conv4').output)
    gap4 = GlobalAveragePooling2D()(base_model.get_layer('block4_conv4').output)
    gap5 = GlobalAveragePooling2D()(base_model.get_layer('block5_conv4').output)

    x = Concatenate()([gap1, gap2, gap3, gap4, gap5, x])

    x = Dense(512, activation='relu', name='fc512_1')(x)
    x = BatchNormalization(name="batch_1")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='fc512_2')(x)
    x = BatchNormalization(name="batch_2")(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='sigmoid', name='fc28')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=x, name='VGG19')

    return model

model = build_model(num_classes=28)
print model.summary()
