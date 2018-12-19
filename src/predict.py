from __future__ import print_function
from __future__ import division

import os
import sys

import argparse
import numpy as np
import importlib

from keras.models import load_model

from configure import *
from utils import load_data, generate_exp_config
from utils import get_weights_path
from utils import get_training_predict_path
from utils import get_test_predict_path
from utils import get_custom_objects
from utils import get_target
from utils import get_data_path
from utils import get_test_time_augmentation_generators


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', "--net_name", type=str, default=None,
                        help='name of convolutional neural network. DEFAULT: None')
    parser.add_argument('-w', "--workers", type=int, default=16, help="number of cores for prediction. DEFAULT: 32")
    parser.add_argument('-v', "--verbose", type=int, default=2, help="Verbosity mode. DEFAULT: 2")
    return parser.parse_args()


def predict_validation(args):
    net = importlib.import_module("Nets." + args.net_name)
    weights_path = get_weights_path(net_name=args.net_name)

    batch_size = net.BATCH_SIZE
    input_shape = net.INPUT_SHAPE

    train_data, _ = get_data_path(input_shape=input_shape)

    print("load training data...", file=sys.stderr)
    print("=======================================================\n", file=sys.stderr)

    img = load_data(data_path=train_data)
    target = get_target()

    training_pred = np.zeros(shape=target.shape, dtype=np.float32)

    for fold in range(K_FOLD):
        exp_config = generate_exp_config(net_name=args.net_name, k_fold=fold)
        weights_filename = os.path.join(weights_path, "{}.h5".format(exp_config))
        assert os.path.exists(weights_filename), "the file: {} doesn't exist...".format(weights_filename)
        custom_objects = get_custom_objects(net_name=args.net_name)
        model = load_model(filepath=weights_filename, custom_objects=custom_objects)

        split_filename = os.path.join(DATA_DIR, "KFold_{}.npz".format(fold))
        split = np.load(file=split_filename)

        valid_indexes = split['test_indexes']

        print("validate the model on {} samples".format(valid_indexes.shape[0]), file=sys.stderr)

        valid_generators = get_test_time_augmentation_generators(image=img,
                                                                 indexes=valid_indexes,
                                                                 batch_size=batch_size,
                                                                 input_shape=input_shape)

        valid_pred = np.zeros(shape=(valid_indexes.shape[0], N_LABELS), dtype=np.float32)

        for valid_generator in valid_generators:
            valid_pred += model.predict_generator(generator=valid_generator,
                                                  use_multiprocessing=True,
                                                  workers=args.workers,
                                                  verbose=args.verbose)

        valid_pred /= len(valid_generators)

        for i, index in enumerate(valid_indexes):
            training_pred[index] = valid_pred[i]

    training_predict_path = get_training_predict_path(net_name=args.net_name)
    filename = os.path.join(training_predict_path, "{}.npz".format(args.net_name))
    np.savez(file=filename, pred=training_pred, label=target)


def predict_test(args):
    net = importlib.import_module("Nets." + args.net_name)
    weights_path = get_weights_path(net_name=args.net_name)

    batch_size = net.BATCH_SIZE
    input_shape = net.INPUT_SHAPE

    _, test_data = get_data_path(input_shape=input_shape)

    print("load test data...", file=sys.stderr)
    print("=======================================================\n", file=sys.stderr)

    img = load_data(data_path=test_data)
    test_generators = get_test_time_augmentation_generators(image=img,
                                                            batch_size=batch_size,
                                                            input_shape=input_shape)

    test_pred = np.zeros(shape=(img.shape[0], N_LABELS), dtype=np.float32)
    for fold in range(1):
        print("predicting for fold {}...\n".format(fold), file=sys.stderr)

        exp_config = generate_exp_config(net_name=args.net_name, k_fold=fold)
        weights_filename = os.path.join(weights_path, "{}.h5".format(exp_config))
        assert os.path.exists(weights_filename), "the file: {} doesn't exist...".format(weights_filename)
        custom_objects = get_custom_objects(net_name=args.net_name)
        model = load_model(filepath=weights_filename, custom_objects=custom_objects)

        for test_generator in test_generators:
            test_pred += model.predict_generator(generator=test_generator,
                                                 use_multiprocessing=True,
                                                 workers=args.workers,
                                                 verbose=args.verbose)

    test_pred /= (1 * len(test_generators))

    test_predict_path = get_test_predict_path(net_name=args.net_name)
    filename = os.path.join(test_predict_path, "{}.npz".format(args.net_name))
    np.savez(file=filename, pred=test_pred)


if __name__ == '__main__':
    arguments = parse_args()
    # predict_validation(args=arguments)
    predict_test(args=arguments)
