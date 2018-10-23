from __future__ import print_function
import os

import argparse

import importlib

from keras.models import load_model
from generator import ImageDataGenerator
from evaluate import evaluate
from predict import predict
from visualization import visua_acc_loss
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_name")
    return parser.parse_args()


def main():
    args = parse_args()

    print("load training and validation data...", file=sys.stderr)

    x_train, y_train = load_data(dataset="train")
    x_valid, y_valid = load_data(dataset="validation")

    exp_id = generate_expid(args.net_name)

    net = importlib.import_module("Nets." + args.net_name)

    # import model parameters
    epochs = net.epochs
    batch_size = net.batch_size
    callbacks = net.build_callbacks(MODEL_PATH, args.net_name)
    input_shape = net.INPUT_SHAPE
    if hasattr(net, 'preprocess_input'):
        preprocessing_function = net.preprocess_input
    else:
        preprocessing_function = None

    weight_filename = os.path.join(MODEL_PATH, "{}.h5".format(args.net_name))
    if os.path.exists(weight_filename):
        model = load_model(weight_filename)
    else:
        model = net.build_model(input_shape=input_shape, num_classes=N_LABELS)

    if net.augment:
        # get augmentation parameters
        augment_parameters = net.augment_parameters
        rotation_range = augment_parameters.get('rotation_range', 0.)
        width_shift_range = augment_parameters.get('width_shift_range', 0.)
        height_shift_range = augment_parameters.get('height_shift_range', 0.)
        brightness_range = augment_parameters.get('brightness_range', 0.)
        shear_range = augment_parameters.get('shear_range', 0.)
        zoom_range = augment_parameters.get('zoom_range', 0.)
        channel_shift_range = augment_parameters.get('channel_shift_range', 0.)
        fill_mode = augment_parameters.get('fill_mode', 'nearest')
        cval = augment_parameters.get('cval', 0.)
        horizontal_flip = augment_parameters.get('horizontal_flip', True)
        vertical_flip = augment_parameters.get('vertical_flip', True)

        train_generator = ImageDataGenerator(x=x_train, y=y_train, batch_size=batch_size,
                                             augment=True, shuffle=True,
                                             output_shape=(input_shape[0], input_shape[1]),
                                             n_channels=input_shape[2],
                                             rotation_range=rotation_range,
                                             width_shift_range=width_shift_range,
                                             height_shift_range=height_shift_range,
                                             brightness_range=brightness_range,
                                             shear_range=shear_range,
                                             zoom_range=zoom_range,
                                             channel_shift_range=channel_shift_range,
                                             fill_mode=fill_mode,
                                             cval=cval,
                                             horizontal_flip=horizontal_flip,
                                             vertical_flip=vertical_flip,
                                             preprocessing_function=preprocessing_function)
    else:
        train_generator = ImageDataGenerator(x=x_train, y=y_train, batch_size=batch_size,
                                             augment=False, shuffle=True,
                                             output_shape=(input_shape[0], input_shape[1]),
                                             n_channels=input_shape[2],
                                             preprocessing_function=preprocessing_function)

    valid_generator = ImageDataGenerator(x=x_valid, y=y_valid, batch_size=batch_size,
                                         output_shape=(input_shape[0], input_shape[1]),
                                         n_channels=input_shape[2],
                                         preprocessing_function=preprocessing_function)

    print("training model...", file=sys.stderr)

    # from utils import SysMonitor
    # sys_mon = SysMonitor()
    # sys_mon.start()

    if net.use_multiprocessing:
        history = model.fit_generator(generator=train_generator,
                                      validation_data=valid_generator,
                                      epochs=epochs,
                                      verbose=2,
                                      callbacks=callbacks,
                                      use_multiprocessing=True,
                                      workers=8)
    else:
        history = model.fit_generator(generator=train_generator,
                                      validation_data=valid_generator,
                                      epochs=epochs,
                                      verbose=2,
                                      callbacks=callbacks)

    # sys_mon.stop()
    # sys_mon.plot(exp_id)
    # exit(0)
    print("training is done!", file=sys.stderr)
    del train_generator, valid_generator

    visua_acc_loss(history, exp_id)

    train_f1, val_f1, optimal_thresholds = evaluate(model, args.net_name, x_train, y_train, x_valid, y_valid,
                                                    output_shape=(input_shape[0], input_shape[1]),
                                                    n_channels=input_shape[2],
                                                    preprocessing_function=preprocessing_function)

    predict(model, args.net_name, train_f1, val_f1,
            output_shape=(input_shape[0], input_shape[1]),
            n_channels=input_shape[2],
            optimal_thresholds=optimal_thresholds,
            preprocessing_function=preprocessing_function)

    print("complete!!")


if __name__ == '__main__':
    main()
