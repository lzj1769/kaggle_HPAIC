from __future__ import print_function

import argparse
import importlib
import sys

from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.metrics import binary_accuracy
from keras.utils.multi_gpu_utils import multi_gpu_model
import keras.backend as K

from callback import build_callbacks
from generator import ImageDataGenerator
from generator import BatchBalanceImageDataGenerator
from utils import *
from configure import *

from albumentations import HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomBrightness


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--net_name", type=str, default=None,
                        help='name of convolutional neural network.')
    parser.add_argument("-k", "--k_fold", type=int, default=0,
                        help="number of KFold split, should between 0 and 1")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="number of epochs for training. DEFAULT: 100")
    parser.add_argument("-g", "--n_gpus", type=int, default=2,
                        help="number of GPUS for training, DEFAULT: 2")
    parser.add_argument("-w", "--workers", type=int, default=8,
                        help="number of cores for training. DEFAULT: All 16 cpus")
    parser.add_argument("-v", "--verbose", type=int, default=2,
                        help="Verbosity mode. DEFAULT: 2")
    parser.add_argument("-l", "--label", type=int, default=None,
                        help="label of training")
    return parser.parse_args()


def main():
    args = parse_args()

    print("load the model configuration...", file=sys.stderr)
    print("===========================================================================\n", file=sys.stderr)

    exp_config = generate_exp_config_single_label(net_name=args.net_name, k_fold=args.k_fold)
    weights_path = get_weights_path(net_name=args.net_name)

    net = importlib.import_module("Nets." + args.net_name)

    batch_size = net.BATCH_SIZE
    input_shape = net.INPUT_SHAPE
    max_queue_size = net.MAX_QUEUE_SIZE
    learning_rate = net.LEARNING_RATE

    weights_filename = os.path.join(weights_path, "{}.h5".format(exp_config))
    model = net.build_model(num_classes=1)

    if os.path.exists(weights_filename):
        model.load_weights(weights_filename, by_name=True)
        optimizer = Adam(lr=learning_rate * 0.1)

    else:
        model.summary()
        optimizer = Adam(lr=learning_rate)

    parallel_model = multi_gpu_model(model=model, gpus=args.n_gpus, cpu_merge=False)

    model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])

    parallel_model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])

    print("load training and validation data...", file=sys.stderr)
    print("===========================================================================\n", file=sys.stderr)

    train_data, _ = get_data_path(input_shape=input_shape)

    img = load_data(data_path=train_data)
    target = get_target()[:, args.label]

    split_filename = os.path.join(DATA_DIR, "Label_{}_KFold_{}.npz".format(args.label, args.k_fold))
    split = np.load(file=split_filename)

    train_indexes = split['train_indexes']
    test_indexes = split['test_indexes']

    print("Training model on {} samples, validate on {} samples".format(len(train_indexes),
                                                                        len(test_indexes),
                                                                        file=sys.stderr))

    # set augmentation parameters
    horizontal_flip = HorizontalFlip(p=0.5)
    vertical_flip = VerticalFlip(p=0.5)
    shift_scale_rotate = ShiftScaleRotate(p=0.8, scale_limit=0.2, rotate_limit=90)
    random_brightness = RandomBrightness(p=0.1, limit=0.1)

    train_generator = BatchBalanceImageDataGenerator(x=img,
                                                     y=target,
                                                     indexes=train_indexes,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     input_shape=input_shape,
                                                     horizontal_flip=horizontal_flip,
                                                     vertical_flip=vertical_flip,
                                                     shift_scale_rotate=shift_scale_rotate,
                                                     random_brightness=random_brightness)

    valid_generator = ImageDataGenerator(x=img,
                                         y=target,
                                         indexes=test_indexes,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         input_shape=input_shape,
                                         learning_phase=True)

    history_path = get_history_path(net_name=args.net_name)
    acc_loss_path = get_acc_loss_path(net_name=args.net_name)
    callbacks = build_callbacks(model=model,
                                weights_path=weights_path,
                                history_path=history_path,
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
                                 max_queue_size=max_queue_size)

    print("complete!!")
    K.clear_session()


if __name__ == '__main__':
    main()