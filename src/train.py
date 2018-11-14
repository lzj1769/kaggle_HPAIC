from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import tensorflow as tf
import importlib
import json

from keras.models import load_model
from keras.losses import binary_crossentropy
from keras import optimizers
from keras.optimizers import SGD
from keras.metrics import binary_accuracy
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.utils.io_utils import h5dict
import keras.backend as K

from generator import ImageDataGenerator
from utils import get_acc_loss_path, load_data, generate_exp_config
from utils import get_weights_path
from utils import get_logs_path, get_custom_objects
from callback import build_callbacks
from configure import *

from albumentations import HorizontalFlip, VerticalFlip
from albumentations import ShiftScaleRotate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_name", type=str, default=None,
                        help='name of convolutional neural network.')
    parser.add_argument("--k_fold", type=int, default=0,
                        help="number of KFold split, should between 0 and 5")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs for training. DEFAULT: 100")
    parser.add_argument("--n_gpus", type=int, default=2,
                        help="number of GPUS for training, DEFAULT: 2")
    parser.add_argument("--workers", type=int, default=32,
                        help="number of cores for training. DEFAULT: 32")
    parser.add_argument("--verbose", type=int, default=2,
                        help="Verbosity mode. DEFAULT: 2")
    return parser.parse_args()


def weighted_binary_corssentropy(y_true, y_pred):
    pos_weights = np.clip(1.0 / np.array(FRACTION) - 1, 1, 10)
    _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    output = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    return K.mean(tf.nn.weighted_cross_entropy_with_logits(y_true, output, pos_weights), axis=-1)


def main():
    args = parse_args()

    print("load the model configuration...", file=sys.stderr)
    print("===========================================================================\n", file=sys.stderr)

    exp_config = generate_exp_config(args.net_name, args.k_fold)
    weights_path = get_weights_path(net_name=args.net_name)

    net = importlib.import_module("Nets." + args.net_name)

    batch_size = net.batch_size
    input_shape = net.input_shape

    # Training models with weights merge on CPU
    with tf.device('/cpu:0'):
        weights_filename = os.path.join(weights_path, "{}.h5".format(exp_config))

        if os.path.exists(weights_filename):
            custom_objects = get_custom_objects(args.net_name)
            f = h5dict(weights_filename, 'r')
            training_config = f.get('training_config')

            training_config = json.loads(training_config.decode('utf-8'))
            optimizer_config = training_config['optimizer_config']
            optimizer = optimizers.deserialize(optimizer_config,
                                               custom_objects=custom_objects)

            model = load_model(weights_filename,
                               custom_objects=custom_objects,
                               compile=False)

            f.close()
        else:
            model = net.build_model(num_classes=N_LABELS)
            optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True, decay=1e-06)

    parallel_model = multi_gpu_model(model=model, gpus=args.n_gpus)

    # model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])
    # parallel_model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])

    model.compile(optimizer=optimizer, loss=weighted_binary_corssentropy, metrics=[binary_accuracy])
    parallel_model.compile(optimizer=optimizer, loss=weighted_binary_corssentropy, metrics=[binary_accuracy])

    model.summary()

    print("load training and validation data...", file=sys.stderr)
    print("===========================================================================\n", file=sys.stderr)

    img, label = load_data(dataset="train")

    split_filename = os.path.join(DATA_DIR, "KFold_{}.npz".format(args.k_fold))
    split = np.load(split_filename)

    train_indexes = split['train_indexes']
    test_indexes = split['test_indexes']

    print("Training model on {} samples, validate on {} samples".format(train_indexes.shape[0],
                                                                        test_indexes.shape[0]), file=sys.stderr)

    # set augmentation parameters
    horizontal_flip = HorizontalFlip(p=0.5)
    vertical_flip = VerticalFlip(p=0.5)
    shift_scale_rotate = ShiftScaleRotate(p=0.5, rotate_limit=90, scale_limit=0.4)

    train_generator = ImageDataGenerator(x=img[train_indexes],
                                         y=label[train_indexes],
                                         batch_size=batch_size,
                                         n_classes=N_LABELS,
                                         shuffle=True,
                                         output_shape=(input_shape[0], input_shape[1]),
                                         n_channels=input_shape[2],
                                         learning_phase=True,
                                         horizontal_flip=horizontal_flip,
                                         vertical_flip=vertical_flip,
                                         shift_scale_rotate=shift_scale_rotate)

    valid_generator = ImageDataGenerator(x=img[test_indexes],
                                         y=label[test_indexes],
                                         batch_size=batch_size,
                                         n_classes=N_LABELS,
                                         shuffle=False,
                                         learning_phase=True,
                                         output_shape=(input_shape[0], input_shape[1]),
                                         n_channels=input_shape[2])

    logs_path = get_logs_path(net_name=args.net_name)
    acc_loss_path = get_acc_loss_path(args.net_name)
    callbacks = build_callbacks(model=model,
                                weights_path=weights_path,
                                logs_path=logs_path,
                                acc_loss_path=acc_loss_path,
                                exp_config=exp_config)

    del img, label
    print("training model...", file=sys.stderr)
    print("===========================================================================\n", file=sys.stderr)
    parallel_model.fit_generator(generator=train_generator,
                                 validation_data=valid_generator,
                                 epochs=args.epochs,
                                 verbose=args.verbose,
                                 callbacks=callbacks,
                                 use_multiprocessing=True,
                                 workers=args.workers)

    print("complete!!")
    K.clear_session()


if __name__ == '__main__':
    main()
