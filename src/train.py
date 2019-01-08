from __future__ import print_function

import argparse
import importlib
import sys

from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.metrics import binary_accuracy
from keras.utils.multi_gpu_utils import multi_gpu_model

from callback import EarlyStoppingWithTime
from callback import CSVPDFLogger
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint

# from callback import build_callbacks
from generator import ImageDataGenerator
from utils import *
from configure import *

from albumentations import HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomBrightness


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--net_name", type=str, default=None,
                        help='name of convolutional neural network.')
    parser.add_argument("-k", "--k_fold", type=int, default=0,
                        help="number of KFold split, should between 0 and 5")
    parser.add_argument("-e", "--epochs", type=int, default=100,
                        help="number of epochs for training. DEFAULT: 30")
    parser.add_argument("-g", "--n_gpus", type=int, default=2,
                        help="number of GPUS for training, DEFAULT: 2")
    parser.add_argument("-w", "--workers", type=int, default=8,
                        help="number of cores for training. DEFAULT: All 16 cpus")
    parser.add_argument("-v", "--verbose", type=int, default=2,
                        help="Verbosity mode. DEFAULT: 2")
    return parser.parse_args()


def build_callbacks(weights_path=None,
                    history_path=None,
                    acc_loss_path=None,
                    exp_config=None):
    check_point_path = os.path.join(weights_path, "{}.h5".format(exp_config))
    history_filename = os.path.join(history_path, "{}.log".format(exp_config))
    pdf_filename = os.path.join(acc_loss_path, "{}.pdf".format(exp_config))

    check_pointer = ModelCheckpoint(filepath=check_point_path,
                                    monitor='val_loss',
                                    save_best_only=True,
                                    verbose=1)

    early_stopper = EarlyStoppingWithTime(seconds=3600 * 110,
                                          monitor='val_loss',
                                          patience=20,
                                          verbose=1,
                                          restore_best_weights=True)

    csv_pdf_logger = CSVPDFLogger(pdf_filename=pdf_filename,
                                  filename=history_filename,
                                  append=True)

    learning_rate = ReduceLROnPlateau(patience=5, min_lr=1e-06, verbose=1)

    callbacks = [check_pointer, early_stopper, csv_pdf_logger, learning_rate]

    return callbacks


def main():
    args = parse_args()

    print("load the model configuration...", file=sys.stderr)
    print("===========================================================================\n", file=sys.stderr)

    exp_config = generate_exp_config(net_name=args.net_name, k_fold=args.k_fold)
    weights_path = get_weights_path(net_name=args.net_name)

    net = importlib.import_module("Nets." + args.net_name)

    batch_size = net.BATCH_SIZE
    input_shape = net.INPUT_SHAPE
    max_queue_size = net.MAX_QUEUE_SIZE
    learning_rate = net.LEARNING_RATE

    weights_filename = os.path.join(weights_path, "{}.h5".format(exp_config))
    model = net.build_model(num_classes=N_LABELS)

    if os.path.exists(weights_filename):
        model.load_weights(weights_filename, by_name=True)
        optimizer = Adam(lr=learning_rate)
    else:
        model.summary()
        optimizer = Adam(lr=learning_rate)

    # parallel_model = multi_gpu_model(model=model, gpus=args.n_gpus, cpu_merge=False)

    model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])

    # parallel_model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])

    print("load training and validation data...", file=sys.stderr)
    print("===========================================================================\n", file=sys.stderr)

    train_data, _ = get_data_path(input_shape=input_shape)

    img = load_data(data_path=train_data)
    target = get_target()

    split_filename = os.path.join(DATA_DIR, "KFold_{}.npz".format(args.k_fold))
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

    train_generator = ImageDataGenerator(x=img,
                                         y=target,
                                         indexes=train_indexes,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         input_shape=input_shape,
                                         learning_phase=True,
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
    callbacks = build_callbacks(weights_path=weights_path,
                                history_path=history_path,
                                acc_loss_path=acc_loss_path,
                                exp_config=exp_config)

    print("training model...", file=sys.stderr)
    print("===========================================================================\n", file=sys.stderr)
    model.fit_generator(generator=train_generator,
                        validation_data=valid_generator,
                        epochs=args.epochs,
                        verbose=args.verbose,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        workers=args.workers,
                        max_queue_size=max_queue_size)

    print("complete!!", file=sys.stdout)
    exit(0)


if __name__ == '__main__':
    main()
