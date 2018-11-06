from __future__ import print_function

import os
import sys
import numpy as np
import argparse

import importlib

from keras.models import load_model
from keras.losses import binary_crossentropy
from keras.optimizers import SGD
from keras.metrics import binary_accuracy

from generator import ImageDataGenerator
from visualization import visua_acc_loss
from utils import get_acc_loss_path, load_data, generate_exp_config
from utils import get_weights_path, get_batch_size, get_input_shape
from utils import get_logs_path
from callback import build_callbacks
from configure import *

from albumentations import HorizontalFlip
from albumentations import RandomBrightness
from albumentations import ShiftScaleRotate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_name", help='name of convolutional neural network', default=None)
    parser.add_argument("--k_fold", type=int, default=0, help="number of KFold split, should between 0 and 7")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs for training. DEFAULT: 100")
    parser.add_argument("--workers", type=int, default=8, help="number of cores for training. DEFAULT: 8")
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity mode. DEFAULT: 2")
    return parser.parse_args()


def main():
    args = parse_args()

    print("load training and validation data...", file=sys.stderr)
    print("===========================================================================\n", file=sys.stderr)

    img, label = load_data(dataset="train")

    split_filename = os.path.join(DATA_DIR, "KFold_{}.npz".format(args.k_fold))
    split = np.load(split_filename)

    train_indexes = split['train_indexes']
    test_indexes = split['test_indexes']

    print("Training model on {} samples, validate on {} samples".format(train_indexes.shape[0],
                                                                        test_indexes.shape[0]), file=sys.stderr)

    print("load the model configuration...", file=sys.stderr)
    print("===========================================================================\n", file=sys.stderr)

    exp_config = generate_exp_config(args.net_name, args.k_fold)
    weights_path = get_weights_path(net_name=args.net_name)

    net = importlib.import_module("Nets." + args.net_name)

    batch_size = get_batch_size(args.net_name)
    input_shape = get_input_shape(args.net_name)

    weights_filename = os.path.join(weights_path, "{}.h5".format(exp_config))

    if os.path.exists(weights_filename):
        model = load_model(weights_filename)
    else:
        model = net.build_model(input_shape=input_shape, num_classes=N_LABELS)

    optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)

    # use binary coressentropy function
    model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])

    model.summary()

    # set augmentation parameters
    horizontal_flip = HorizontalFlip(p=0.5)
    shift_scale_rotate = ShiftScaleRotate(p=0.5, rotate_limit=90, scale_limit=0.4)
    random_brightness = RandomBrightness(p=0.2, limit=0.2)

    train_generator = ImageDataGenerator(x=img[train_indexes],
                                         y=label[train_indexes],
                                         batch_size=batch_size,
                                         n_classes=N_LABELS,
                                         shuffle=True,
                                         output_shape=(input_shape[0], input_shape[1]),
                                         n_channels=input_shape[2],
                                         horizontal_flip=horizontal_flip,
                                         shift_scale_rotate=shift_scale_rotate,
                                         random_brightness=random_brightness)

    valid_generator = ImageDataGenerator(x=img[test_indexes],
                                         y=label[test_indexes],
                                         batch_size=batch_size,
                                         n_classes=N_LABELS,
                                         shuffle=False,
                                         output_shape=(input_shape[0], input_shape[1]),
                                         n_channels=input_shape[2])

    logs_path = get_logs_path(net_name=args.net_name)
    callbacks = build_callbacks(weights_path=weights_path, logs_path=logs_path, exp_config=exp_config)

    del img, label
    print("training model...", file=sys.stderr)
    print("===========================================================================\n", file=sys.stderr)
    model.fit_generator(generator=train_generator,
                        validation_data=valid_generator,
                        epochs=args.epochs,
                        verbose=args.verbose,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        workers=args.workers)

    print("training is done!", file=sys.stderr)

    acc_loss_path = get_acc_loss_path(args.net_name)
    visua_acc_loss(acc_loss_path=acc_loss_path, logs_path=logs_path, exp_config=exp_config)

    print("complete!!")


if __name__ == '__main__':
    main()
