from __future__ import print_function
from __future__ import division

import os
import sys

import argparse
import importlib

import numpy as np

from keras.models import load_model

from generator import ImageDataGenerator
from configure import *
from utils import load_data, generate_exp_config
from utils import get_weights_path, get_batch_size, get_input_shape
from utils import get_training_predict_path, get_test_predict_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_name", help='name of convolutional neural network', default=None)
    parser.add_argument("--pre_trained", type=int, default=1,
                        help="whether use the pre-trained weights or not, set 0 will train the network from "
                             "scratch and 1 will use the weights from imagenet. DEFAULT: 1")
    parser.add_argument("--include_fc", type=int, default=0,
                        help="whether include the full connect layers for trained neural network. DEFAULT 0")
    parser.add_argument("--k_fold", type=int, default=0, help="number of KFold split, should between 0 and 7")
    parser.add_argument("--workers", type=int, default=8, help="number of cores for training. DEFAULT: 8")
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity mode. DEFAULT: 2")
    return parser.parse_args()


def main():
    args = parse_args()

    print("load the model configuration...", file=sys.stderr)
    print("=======================================================", file=sys.stderr)

    exp_config = generate_exp_config(args.net_name, args.pre_trained, args.include_fc, args.k_fold)
    weights_path = get_weights_path(net_name=args.net_name)

    net = importlib.import_module("Nets." + args.net_name)

    batch_size = get_batch_size(args.net_name, args.pre_trained)
    input_shape = get_input_shape(args.net_name, args.pre_trained)

    if args.pre_trained:
        preprocessing_function = net.preprocess_input
    else:
        preprocessing_function = None

    weights_filename = os.path.join(weights_path, "{}.h5".format(exp_config))

    assert os.path.exists(weights_filename), print("the model doesn't exist...", file=sys.stderr)
    model = load_model(weights_filename)

    rotation_range = AUGMENT_PARAMETERS.get('rotation_range', 0.)
    width_shift_range = AUGMENT_PARAMETERS.get('width_shift_range', 0.)
    height_shift_range = AUGMENT_PARAMETERS.get('height_shift_range', 0.)
    shear_range = AUGMENT_PARAMETERS.get('shear_range', 0.)
    zoom_range = AUGMENT_PARAMETERS.get('zoom_range', 0.)
    fill_mode = AUGMENT_PARAMETERS.get('fill_mode', 'nearest')
    cval = AUGMENT_PARAMETERS.get('cval', 0.)
    horizontal_flip = AUGMENT_PARAMETERS.get('horizontal_flip', True)
    vertical_flip = AUGMENT_PARAMETERS.get('vertical_flip', True)

    # output path
    training_predict_path = get_training_predict_path(args.net_name)
    test_predict_path = get_test_predict_path(args.net_name)

    print("load training data...", file=sys.stderr)
    print("=======================================================", file=sys.stderr)

    img, label = load_data(dataset="train")

    split_filename = os.path.join(DATA_DIR, "KFold_{}.npz".format(args.k_fold))
    split = np.load(split_filename)

    test_indexes = split['test_indexes']

    print("validate the model on {} samples".format(test_indexes.shape[0]), file=sys.stderr)

    valid_generator = ImageDataGenerator(x=img[test_indexes], y=None,
                                         batch_size=batch_size,
                                         augment=False, shuffle=False,
                                         output_shape=(input_shape[0], input_shape[1]),
                                         n_channels=input_shape[2],
                                         preprocessing_function=preprocessing_function)

    valid_generator_aug = ImageDataGenerator(x=img[test_indexes], y=None,
                                             batch_size=batch_size,
                                             augment=True, shuffle=False,
                                             output_shape=(input_shape[0], input_shape[1]),
                                             n_channels=input_shape[2],
                                             rotation_range=rotation_range,
                                             width_shift_range=width_shift_range,
                                             height_shift_range=height_shift_range,
                                             shear_range=shear_range,
                                             zoom_range=zoom_range,
                                             fill_mode=fill_mode,
                                             cval=cval,
                                             horizontal_flip=horizontal_flip,
                                             vertical_flip=vertical_flip,
                                             preprocessing_function=preprocessing_function,
                                             augment_prob=1.0)

    valid_pred = model.predict_generator(valid_generator, use_multiprocessing=True, workers=8)
    valid_pred_aug = np.zeros((test_indexes.shape[0], N_LABELS), dtype=np.float32)
    for i in range(TEST_TIME_AUGMENT):
        valid_pred_aug += model.predict_generator(valid_generator_aug, use_multiprocessing=True, workers=8)

    valid_pred = 0.5 * valid_pred + 0.5 * valid_pred_aug / TEST_TIME_AUGMENT

    filename = os.path.join(training_predict_path, "{}.npz".format(exp_config))
    np.savez(file=filename, pred=valid_pred, label=label[test_indexes])

    print("load test data...", file=sys.stderr)
    print("=======================================================", file=sys.stderr)

    x_test = load_data(dataset="test")

    test_generator = ImageDataGenerator(x=x_test, batch_size=batch_size,
                                        augment=False, shuffle=False,
                                        output_shape=(input_shape[0], input_shape[1]),
                                        n_channels=input_shape[2],
                                        preprocessing_function=preprocessing_function)

    test_generator_aug = ImageDataGenerator(x=x_test,
                                            batch_size=batch_size,
                                            augment=True, shuffle=False,
                                            output_shape=(input_shape[0], input_shape[1]),
                                            n_channels=input_shape[2],
                                            rotation_range=rotation_range,
                                            width_shift_range=width_shift_range,
                                            height_shift_range=height_shift_range,
                                            shear_range=shear_range,
                                            zoom_range=zoom_range,
                                            fill_mode=fill_mode,
                                            cval=cval,
                                            horizontal_flip=horizontal_flip,
                                            vertical_flip=vertical_flip,
                                            preprocessing_function=preprocessing_function,
                                            augment_prob=1.0)

    test_pred = model.predict_generator(test_generator, use_multiprocessing=True, workers=8)
    test_pred_aug = np.zeros((x_test.shape[0], N_LABELS), dtype=np.float32)
    for i in range(TEST_TIME_AUGMENT):
        test_pred_aug += model.predict_generator(test_generator_aug, use_multiprocessing=True, workers=8)

    test_pred = 0.5 * test_pred + 0.5 * test_pred_aug / TEST_TIME_AUGMENT

    filename = os.path.join(test_predict_path, "{}.npz".format(exp_config))
    np.savez(file=filename, pred=test_pred)


if __name__ == '__main__':
    main()
