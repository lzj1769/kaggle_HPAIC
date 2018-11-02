from __future__ import print_function

import argparse

import importlib

from keras.models import load_model
from keras.losses import binary_crossentropy
from keras.optimizers import Adamax, Adam
from keras.metrics import binary_accuracy
from loss import focal_loss, f1_loss

from generator import ImageDataGenerator
from visualization import visua_acc_loss
from utils import *
from callback import build_callbacks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_name", help='name of convolutional neural network', default=None)
    parser.add_argument("--pre_trained", type=int, default=1,
                        help="whether use the pre-trained weights or not, set 0 will train the network from "
                             "scratch and 1 will use the weights from imagenet. DEFAULT: 1")
    parser.add_argument("--optimizer", type=int, default=0,
                        help="which optimizer should use to train neural network. DEFAULT 0")
    parser.add_argument("--loss", type=int, default=0,
                        help="which loss function should be used to train the model")
    parser.add_argument("--k_fold", type=int, default=0, help="number of KFold split, should between 0 and 7")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs for training. DEFAULT: 100")
    parser.add_argument("--workers", type=int, default=8, help="number of cores for training. DEFAULT: 8")
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity mode. DEFAULT: 2")
    return parser.parse_args()


def main():
    args = parse_args()

    print("load the model configuration...", file=sys.stderr)
    print("=======================================================", file=sys.stderr)

    exp_config = generate_exp_config(args.net_name, args.pre_trained, args.loss, args.k_fold)
    weights_path = get_weights_path(net_name=args.net_name)

    net = importlib.import_module("Nets." + args.net_name)

    batch_size = get_batch_size(args.net_name, args.pre_trained)
    input_shape = get_input_shape(args.net_name, args.pre_trained)

    if args.pre_trained:
        preprocessing_function = net.preprocess_input
    else:
        preprocessing_function = None

    weights_filename = os.path.join(weights_path, "{}.h5".format(exp_config))
    if os.path.exists(weights_filename):
        model = load_model(weights_filename)
    else:
        if args.pre_trained:
            model = net.build_model(input_shape=input_shape, num_classes=N_LABELS,
                                    weights='imagenet')
        else:
            model = net.build_model(input_shape=input_shape, num_classes=N_LABELS,
                                    weights=None)

    # setup the optimizer
    optimizer = None
    if args.pre_trained == 1:
        optimizer = Adam(lr=0.0001, amsgrad=True, decay=1e-06)

    elif args.pre_trained == 0:
        optimizer = Adamax(decay=1e-06)

    # setup the loss function, 0 for binary crossentropy loss, 1 for focal loss, 2 for f1 score loss
    if args.loss == 0:
        model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])
    elif args.loss == 1:
        model.compile(optimizer=optimizer, loss=focal_loss, metrics=[binary_accuracy])
    elif args.loss == 2:
        model.compile(optimizer=optimizer, loss=f1_loss, metrics=[binary_accuracy])

    model.summary()

    print("load training and validation data...", file=sys.stderr)
    print("===========================================================================", file=sys.stderr)

    img, label = load_data(dataset="train")

    split_filename = os.path.join(DATA_DIR, "KFold_{}.npz".format(args.k_fold))
    split = np.load(split_filename)

    train_indexes = split['train_indexes']
    test_indexes = split['test_indexes']

    print("Training model on {} samples, validate on {} samples".format(train_indexes.shape[0],
                                                                        test_indexes.shape[0]), file=sys.stderr)

    # get augmentation parameters
    rotation_range = AUGMENT_PARAMETERS.get('rotation_range', 0.)
    width_shift_range = AUGMENT_PARAMETERS.get('width_shift_range', 0.)
    height_shift_range = AUGMENT_PARAMETERS.get('height_shift_range', 0.)
    shear_range = AUGMENT_PARAMETERS.get('shear_range', 0.)
    zoom_range = AUGMENT_PARAMETERS.get('zoom_range', 0.)
    fill_mode = AUGMENT_PARAMETERS.get('fill_mode', 'nearest')
    cval = AUGMENT_PARAMETERS.get('cval', 0.)
    horizontal_flip = AUGMENT_PARAMETERS.get('horizontal_flip', True)
    vertical_flip = AUGMENT_PARAMETERS.get('vertical_flip', True)

    train_generator = ImageDataGenerator(x=img[train_indexes], y=label[train_indexes],
                                         batch_size=batch_size,
                                         augment=True, shuffle=True,
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
                                         preprocessing_function=preprocessing_function)

    valid_generator = ImageDataGenerator(x=img[test_indexes], y=label[test_indexes],
                                         batch_size=batch_size,
                                         augment=False, shuffle=False,
                                         output_shape=(input_shape[0], input_shape[1]),
                                         n_channels=input_shape[2],
                                         preprocessing_function=preprocessing_function)

    logs_path = get_logs_path(net_name=args.net_name)
    callbacks = build_callbacks(weights_path=weights_path, logs_path=logs_path, exp_config=exp_config)

    del img, label
    print("training model...", file=sys.stderr)
    print("===========================================================================", file=sys.stderr)
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
