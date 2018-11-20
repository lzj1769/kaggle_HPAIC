from __future__ import print_function

import argparse
import tensorflow as tf
import importlib
import json
import sys

from keras.models import load_model
from keras.losses import binary_crossentropy
from keras import optimizers
from keras.optimizers import SGD
from keras.metrics import binary_accuracy
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.utils.io_utils import h5dict
import keras.backend as K

from generator import ImageDataGenerator
from callback import build_callbacks
from utils import *
from configure import *

from albumentations import HorizontalFlip, ShiftScaleRotate


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
                        help="number of cores for training. DEFAULT: All cpus")
    parser.add_argument("--verbose", type=int, default=2,
                        help="Verbosity mode. DEFAULT: 2")
    return parser.parse_args()


def main():
    args = parse_args()

    print("load the model configuration...", file=sys.stderr)
    print("===========================================================================\n", file=sys.stderr)

    exp_config = generate_exp_config(args.net_name, args.k_fold)
    weights_path = get_weights_path(net_name=args.net_name)

    net = importlib.import_module("Nets." + args.net_name)

    batch_size = net.BATCH_SIZE
    input_shape = net.INPUT_SHAPE
    train_data = net.TRAINING_DATA

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
            optimizer = SGD(lr=1e-02, momentum=0.9, decay=1e-04, nesterov=True)

    parallel_model = multi_gpu_model(model=model, gpus=args.n_gpus)

    model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])
    parallel_model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])

    # model.summary()

    print("load training and validation data...", file=sys.stderr)
    print("===========================================================================\n", file=sys.stderr)

    img = load_data(data_path=train_data, image_size=(input_shape[0], input_shape[1]))
    target = get_target()

    split_filename = os.path.join(DATA_DIR, "KFold_{}.npz".format(args.k_fold))
    split = np.load(split_filename)

    train_indexes = split['train_indexes']
    test_indexes = split['test_indexes']

    print("Training model on {} samples, validate on {} samples".format(len(train_indexes),
                                                                        len(test_indexes),
                                                                        file=sys.stderr))

    # set augmentation parameters
    horizontal_flip = HorizontalFlip(p=0.5)
    shift_scale_rotate = ShiftScaleRotate(p=0.8, scale_limit=0.2, rotate_limit=90)

    train_generator = ImageDataGenerator(x=img,
                                         y=target,
                                         indexes=train_indexes,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         input_shape=input_shape,
                                         learning_phase=True,
                                         horizontal_flip=horizontal_flip,
                                         shift_scale_rotate=shift_scale_rotate)

    valid_generator = ImageDataGenerator(x=img,
                                         y=target,
                                         indexes=test_indexes,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         input_shape=input_shape,
                                         learning_phase=True)

    logs_path = get_logs_path(net_name=args.net_name)
    acc_loss_path = get_acc_loss_path(args.net_name)
    callbacks = build_callbacks(model=model,
                                weights_path=weights_path,
                                logs_path=logs_path,
                                acc_loss_path=acc_loss_path,
                                exp_config=exp_config)

    print("training model...", file=sys.stderr)
    print("===========================================================================\n", file=sys.stderr)
    parallel_model.fit_generator(generator=train_generator,
                                 validation_data=valid_generator,
                                 epochs=args.epochs,
                                 verbose=args.verbose,
                                 callbacks=callbacks,
                                 use_multiprocessing=True,
                                 workers=args.workers,
                                 max_queue_size=20)

    print("complete!!")
    K.clear_session()


if __name__ == '__main__':
    main()
