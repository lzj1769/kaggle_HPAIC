from __future__ import print_function
from __future__ import division

import os
import sys

import argparse
import importlib

import pandas as pd
import numpy as np

from keras.models import load_model

from generator import ImageDataGenerator
from configure import *
from utils import load_data, calculate_threshold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_name")
    return parser.parse_args()


def main():
    args = parse_args()

    print("load test data...", file=sys.stderr)

    x_test = load_data(dataset="test")

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

    test_generator = ImageDataGenerator(x=x_test, batch_size=batch_size,
                                        augment=False, shuffle=False,
                                        output_shape=(input_shape[0], input_shape[1]),
                                        n_channels=input_shape[2],
                                        preprocessing_function=preprocessing_function)

    test_generator_aug = ImageDataGenerator(x=x_test, batch_size=batch_size,
                                            augment=True, shuffle=False,
                                            output_shape=(input_shape[0], input_shape[1]),
                                            n_channels=input_shape[2],
                                            preprocessing_function=preprocessing_function,
                                            rotation_range=90,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.2,
                                            zoom_range=0.4,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            augment_prob=1.0)

    test_pred_1 = model.predict_generator(test_generator)
    test_pred_2 = model.predict_generator(test_generator_aug, use_multiprocessing=True, workers=8)
    test_pred_3 = model.predict_generator(test_generator_aug, use_multiprocessing=True, workers=8)

    test_pred = 0.5 * test_pred_1 + 0.5 * ((test_pred_2 + test_pred_3) / 2)

    test_thres = calculate_threshold(test_pred)

    print(test_thres)

    output_test_labels = list()

    # convert the predicted probabilities into labels for training data
    for i in range(test_pred.shape[0]):
        label_predict = np.arange(N_LABELS)[np.greater(test_pred[i], test_thres)]
        if label_predict.size == 0:
            label_predict = [np.argmax(test_pred[i])]

        str_predict_label = " ".join(str(label) for label in label_predict)
        output_test_labels.append(str_predict_label)

    df = pd.read_csv(SAMPLE_SUBMISSION)
    df['Predicted'] = output_test_labels

    submission_filename = "{}_.csv".format(args.net_name)
    filename = os.path.join(SUBMISSION_PATH, submission_filename)
    df.to_csv(filename, index=False)


if __name__ == '__main__':
    main()
