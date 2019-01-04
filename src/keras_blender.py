from __future__ import print_function, division

import os
import sys

import numpy as np
import argparse
from keras import layers
from keras import Model
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from utils import get_training_predict_path
from utils import get_test_predict_path
from utils import get_target
from configure import *

NET_NAMES = ['ResNet50', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'InceptionV3', 'InceptionResNetV2']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", type=int, default=2, help="Verbosity mode. DEFAULT: 2")
    return parser.parse_args()


def get_data():
    x_train = np.empty(shape=(N_TRAINING, N_LABELS, len(NET_NAMES), 1), dtype=np.float32)
    x_test = np.empty(shape=(N_TEST, N_LABELS, len(NET_NAMES), 1), dtype=np.float32)

    for i, net_name in enumerate(NET_NAMES):
        training_predicted_path = get_training_predict_path(net_name)
        filename = os.path.join(training_predicted_path, "{}.npz".format(net_name))
        assert os.path.exists(filename), "the prediction {} does not exist".format(filename)

        x_train[:, :, i, 0] = np.load(filename)['pred']

        test_predicted_path = get_test_predict_path(net_name)
        filename = os.path.join(test_predicted_path, "{}.npz".format(net_name))
        assert os.path.exists(filename), "the prediction {} does not exist".format(filename)

        x_test[:, :, i, 0] = np.load(filename)['pred']

    y_train = get_target()

    return x_train, y_train, x_test


def build_model(input_shape, num_classes):
    input_data = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=8, kernel_size=(3, 1), strides=(1, 1))(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=16, kernel_size=(3, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x = layers.Dense(num_classes, activation='sigmoid', name='fc28')(x)

    model = Model(inputs=input_data, outputs=x, name='Keras_blender')

    return model


def main():
    x_train, y_train, x_test = get_data()

    print(x_train[0])

    valid_prediction = np.zeros(shape=y_train.shape, dtype=np.float32)
    test_validation = np.zeros(shape=(N_TEST, N_LABELS), dtype=np.float32)

    for fold in range(K_FOLD):
        split_filename = os.path.join(DATA_DIR, "KFold_{}.npz".format(fold))
        split = np.load(file=split_filename)

        train_indexes = split['train_indexes']
        test_indexes = split['test_indexes']

        print("Training model on {} samples, validate on {} samples".format(len(train_indexes),
                                                                            len(test_indexes),
                                                                            file=sys.stderr))

        model = build_model(input_shape=(N_LABELS, len(NET_NAMES), 1), num_classes=N_LABELS)

        optimizer = Adam(lr=0.0001)
        model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])

        model.summary()

        check_point_path = os.path.join("/home/rs619065/HPAIC/model/Keras", "KFOLD_{}.h5".format(fold))
        check_pointer = ModelCheckpoint(filepath=check_point_path,
                                        monitor='val_loss',
                                        save_best_only=True,
                                        verbose=1)

        learning_rate = ReduceLROnPlateau(patience=2, min_lr=1e-08, verbose=1)

        early_stopper = EarlyStopping(patience=10, verbose=1, restore_best_weights=True)

        model.fit(x=x_train[train_indexes],
                  y=y_train[train_indexes],
                  batch_size=512,
                  epochs=100,
                  validation_data=[x_train[test_indexes], y_train[test_indexes]],
                  shuffle=True,
                  verbose=1,
                  callbacks=[check_pointer, learning_rate, early_stopper])

        valid_pred = model.predict(x=x_train[test_indexes])

        for i, index in enumerate(test_indexes):
            valid_prediction[index] = valid_pred[i]

        test_pred = model.predict(x=x_test)

        test_validation += test_pred

    filename = "/home/rs619065/HPAIC/training/Keras/Keras.npz"
    np.savez(filename, pred=valid_prediction, label=y_train)

    filename = "/home/rs619065/HPAIC/test/Keras/Keras.npz"
    test_validation /= K_FOLD
    np.savez(filename, pred=test_validation)


if __name__ == '__main__':
    main()
