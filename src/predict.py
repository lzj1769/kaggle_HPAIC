from __future__ import print_function
from __future__ import division

import os
import sys

import argparse
import numpy as np

from keras.models import load_model

from configure import *
from utils import load_data, generate_exp_config
from utils import get_weights_path, get_batch_size, get_input_shape
from utils import get_test_time_augmentation_generators
from utils import get_training_predict_path
from utils import get_test_predict_path

parser = argparse.ArgumentParser()
parser.add_argument("--net_name", help='name of convolutional neural network', default=None)
parser.add_argument("--workers", type=int, default=8, help="number of cores for training. DEFAULT: 8")
parser.add_argument("--verbose", type=int, default=2, help="Verbosity mode. DEFAULT: 2")
args = parser.parse_args()

print("load the model configuration...", file=sys.stderr)
print("=======================================================\n", file=sys.stderr)

batch_size = get_batch_size(args.net_name)
input_shape = get_input_shape(args.net_name)

print("load training data...", file=sys.stderr)
print("=======================================================\n", file=sys.stderr)

img, label = load_data(dataset="train")

weights_path = get_weights_path(net_name=args.net_name)

training_pred = np.zeros((N_TRAINING, N_LABELS), dtype=np.float32)
for fold in range(K_FOLD):
    exp_config = generate_exp_config(args.net_name, fold)
    weights_filename = os.path.join(weights_path, "{}.h5".format(exp_config))
    assert os.path.exists(weights_filename), "the file: {} doesn't exist...".format(weights_filename)
    model = load_model(weights_filename)

    split_filename = os.path.join(DATA_DIR, "KFold_{}.npz".format(fold))
    split = np.load(split_filename)

    valid_indexes = split['test_indexes']

    print("validate the model on {} samples".format(valid_indexes.shape[0]), file=sys.stderr)
    valid_generators = get_test_time_augmentation_generators(image=img[valid_indexes],
                                                             batch_size=batch_size,
                                                             output_shape=(input_shape[0], input_shape[1]),
                                                             n_channels=input_shape[2])

    valid_pred = np.zeros((valid_indexes.shape[0], N_LABELS), dtype=np.float32)
    for generator in valid_generators:
        valid_pred += model.predict_generator(generator,
                                              use_multiprocessing=True,
                                              workers=args.workers)

    valid_pred /= len(valid_generators)
    for i, index in enumerate(valid_indexes):
        training_pred[index] = valid_pred[i]

training_predict_path = get_training_predict_path(args.net_name)
filename = os.path.join(training_predict_path, "{}.npz".format(args.net_name))
np.savez(file=filename, pred=training_pred, label=label)

del img, label

print("load test data...", file=sys.stderr)
print("=======================================================\n", file=sys.stderr)

x_test = load_data(dataset="test")
test_generators = get_test_time_augmentation_generators(image=x_test,
                                                        batch_size=batch_size,
                                                        output_shape=(input_shape[0], input_shape[1]),
                                                        n_channels=input_shape[2])

test_pred = np.zeros((x_test.shape[0], N_LABELS), dtype=np.float32)
for fold in range(K_FOLD):
    print("predicting for fold {}...\n".format(fold), file=sys.stderr)

    exp_config = generate_exp_config(args.net_name, fold)
    weights_filename = os.path.join(weights_path, "{}.h5".format(exp_config))
    assert os.path.exists(weights_filename), "the file: {} doesn't exist...".format(weights_filename)
    model = load_model(weights_filename)

    for generator in test_generators:
        test_pred += model.predict_generator(generator,
                                             use_multiprocessing=True,
                                             workers=args.workers)

test_pred /= (K_FOLD * len(test_generators))

test_predict_path = get_test_predict_path(args.net_name)
filename = os.path.join(test_predict_path, "{}.npz".format(args.net_name))
np.savez(file=filename, pred=test_pred)
