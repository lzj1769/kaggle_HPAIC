# based on https://github.com/Goldesel23/Siamese-Networks-for-One-Shot-Learning/blob/master/siamese_network.py

from __future__ import print_function

from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Lambda
import keras.backend as K

BATCH_SIZE = 16
INPUT_SHAPE = (512, 512, 3)
MAX_QUEUE_SIZE = 32
LEARNING_RATE = 1e-04


def build_model():
    # Let's define the cnn architecture
    net = Sequential()
    net.add(Conv2D(filters=64, kernel_size=(10, 10), activation='relu', input_shape=INPUT_SHAPE))
    net.add(MaxPool2D())

    net.add(Conv2D(filters=128, kernel_size=(7, 7), activation='relu'))
    net.add(MaxPool2D())

    net.add(Conv2D(filters=128, kernel_size=(4, 4), activation='relu'))
    net.add(MaxPool2D())

    net.add(Conv2D(filters=256, kernel_size=(4, 4), activation='relu'))
    net.add(MaxPool2D())

    net.add(Flatten())
    net.add(Dense(units=1024, activation='sigmoid', name='Dense1'))

    # Now the pairs of images
    input_image_1 = Input(INPUT_SHAPE)
    input_image_2 = Input(INPUT_SHAPE)

    encoded_image_1 = net(input_image_1)
    encoded_image_2 = net(input_image_2)

    # L1 distance layer between the two encoded outputs
    # One could use Subtract from Keras, but we want the absolute value
    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    l1_distance = l1_distance_layer([encoded_image_1, encoded_image_2])

    # Same class or not prediction
    prediction = Dense(units=1, activation='sigmoid')(l1_distance)
    model = Model(inputs=[input_image_1, input_image_2], outputs=prediction)

    return model
