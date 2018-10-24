from __future__ import print_function
from __future__ import division

import os
import sys

import pandas as pd
import numpy as np
import argparse
import importlib

from keras.models import load_model

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from generator import ImageDataGenerator
from configure import *
from utils import load_data
from visualization import visua_threshold_f1, visua_f1_classes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_name")
    return parser.parse_args()


def main():

    args = parse_args()

    print("load training and validation data...", file=sys.stderr)

    x_train, y_train = load_data(dataset="train")
    x_valid, y_valid = load_data(dataset="validation")

    net = importlib.import_module("Nets." + args.net_name)

    # import model parameters
    batch_size = net.batch_size
    input_shape = net.INPUT_SHAPE
    if hasattr(net, 'preprocess_input'):
        preprocessing_function = net.preprocess_input
    else:
        preprocessing_function = None

    print("load model...", file=sys.stderr)
    weight_filename = os.path.join(MODEL_PATH, "{}.h5".format(args.net_name))
    assert os.path.exists(weight_filename), print("The model does not exist...")
    model = load_model(weight_filename)

    print("evaluating the model...", file=sys.stderr)
    train_generator = ImageDataGenerator(x=x_train, y=y_train, batch_size=batch_size,
                                         augment=False, shuffle=False,
                                         output_shape=(input_shape[0], input_shape[1]),
                                         n_channels=input_shape[2],
                                         preprocessing_function=preprocessing_function)

    valid_generator = ImageDataGenerator(x=x_valid, y=y_valid, batch_size=batch_size,
                                         augment=False, shuffle=False,
                                         output_shape=(input_shape[0], input_shape[1]),
                                         n_channels=input_shape[2],
                                         preprocessing_function=preprocessing_function)

    train_pred = model.predict_generator(train_generator)
    valid_pred = model.predict_generator(valid_generator)



def evaluate(model, exp_id, x_train, y_train, x_valid, y_valid,
             output_shape, n_channels, preprocessing_function):
    print("evaluating the model...", file=sys.stderr)

    # evaluate our model on test and validation data
    train_generator = ImageDataGenerator(x=x_train, output_shape=output_shape,
                                         shuffle=False, augment=False,
                                         n_channels=n_channels,
                                         preprocessing_function=preprocessing_function)

    valid_generator = ImageDataGenerator(x=x_valid, output_shape=output_shape,
                                         shuffle=False, augment=False,
                                         n_channels=n_channels,
                                         preprocessing_function=preprocessing_function)

    train_pred = model.predict_generator(train_generator)
    valid_pred = model.predict_generator(valid_generator)


    # visualize the f1 score for each class based on different threshold
    visua_threshold_f1(f1, optimal_thresholds, exp_id)
    visua_f1_classes(optimal_f1_score, exp_id)

    mlb = MultiLabelBinarizer(classes=range(N_LABELS))

    train_labels = list()
    valid_labels = list()
    output_train_labels = list()
    output_valid_labels = list()

    # convert the predicted probabilities into labels for training data
    for i in range(train_pred.shape[0]):
        label_predict = np.arange(N_LABELS)[np.greater(train_pred[i], optimal_thresholds)]
        if label_predict.size == 0:
            train_labels.append(np.arange(N_LABELS))
            str_predict_label = " ".join(str(label) for label in np.arange(N_LABELS))
        else:
            train_labels.append(label_predict)
            str_predict_label = " ".join(str(label) for label in label_predict)

        output_train_labels.append(str_predict_label)

    # convert the predicted probabilities into labels for validation data
    for i in range(valid_pred.shape[0]):
        label_predict = np.arange(N_LABELS)[np.greater(valid_pred[i], optimal_thresholds)]
        if label_predict.size == 0:
            valid_labels.append(np.arange(N_LABELS))
            str_predict_label = " ".join(str(label) for label in np.arange(N_LABELS))
        else:
            str_predict_label = " ".join(str(label) for label in label_predict)
            valid_labels.append(label_predict)

        output_valid_labels.append(str_predict_label)

    train_labels_bin = mlb.fit_transform(train_labels)
    valid_labels_bin = mlb.fit_transform(valid_labels)

    train_f1 = f1_score(y_true=y_train, y_pred=train_labels_bin, average='macro').round(3)
    valid_f1 = f1_score(y_true=y_valid, y_pred=valid_labels_bin, average='macro').round(3)

    print("evaluation is done, f1 score of training: {}, f1 score of validation: {}".format(train_f1, valid_f1))

    df_train = pd.read_csv(TRAINING_DATA_CSV)
    df_train['Predicted'] = output_train_labels

    for i in range(train_pred.shape[1]):
        res = list()
        for j in range(train_pred.shape[0]):
            res.append(str(train_pred[j, i]))
        df_train[str(i)] = res

    train_filename = "{}_train_f1_{}_val_f1_{}.csv".format(exp_id, train_f1, valid_f1)
    filename = os.path.join(TRAINING_PATH, train_filename)
    df_train.to_csv(filename, index=False)

    df_valid = pd.read_csv(VALIDATION_DATA_CSV)
    df_valid['Predicted'] = output_valid_labels

    for i in range(valid_pred.shape[1]):
        res = list()
        for j in range(valid_pred.shape[0]):
            res.append(str(valid_pred[j, i]))
        df_valid[str(i)] = res

    validation_filename = "{}_train_f1_{}_val_f1_{}.csv".format(exp_id, train_f1, valid_f1)
    filename = os.path.join(VALIDATION_PATH, validation_filename)
    df_valid.to_csv(filename, index=False)

    return train_f1, valid_f1, optimal_thresholds


if __name__ == '__main__':
    main()
