import sys

from keras import layers
from keras import Model
from keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Concatenate

sys.setrecursionlimit(3000)

BATCH_SIZE = 16
INPUT_SHAPE = (1024, 1024, 3)
MAX_QUEUE_SIZE = 32
LEARNING_RATE = 1e-04


def Xception(input_shape=INPUT_SHAPE):
    """Instantiates the Xception architecture.

    """
    # Determine proper input shape
    img_input = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3),
                      strides=(2, 2),
                      use_bias=False,
                      name='block1_conv1')(img_input)

    x = layers.BatchNormalization(name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    residual = layers.Conv2D(128, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv2')(x)
    x = layers.BatchNormalization(name='block2_sepconv2_bn')(x)
    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)

    x = layers.add([x, residual])

    residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)

    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv2')(x)
    x = layers.BatchNormalization(name='block3_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(512, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv1')(x)
    x = layers.BatchNormalization(name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv1_act')(x)

    x = layers.GlobalAveragePooling2D()(x)

    # Create model.
    model = Model(img_input, x, name='xception')

    return model


def build_model(num_classes):
    # create the base pre-trained model
    base_model = Xception()

    # add a global spatial average pooling layer
    x = base_model.output

    gap1 = GlobalAveragePooling2D()(base_model.get_layer('block2_pool').output)
    gap2 = GlobalAveragePooling2D()(base_model.get_layer('block3_pool').output)

    x = Concatenate()([gap1, gap2, x])

    x = Dense(512, activation='relu', name='fc1')(x)
    x = BatchNormalization(name="batch_1")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = BatchNormalization(name="batch_2")(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='sigmoid', name='fc28')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=x, name='Xception')

    return model
