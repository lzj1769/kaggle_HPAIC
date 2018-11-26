from __future__ import print_function, division

import os

import pandas as pd
import numpy as np
from albumentations import ShiftScaleRotate, HorizontalFlip
from PIL import Image
from configure import *
from generator import ImageDataGenerator
from utils import load_data, get_test_time_augmentation_generators
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--train", action='store_true', default=False)
    parser.add_argument("--tta", action='store_true', default=False)
    return parser.parse_args()


args = parse_args()

if args.test:
    horizontal_flip = HorizontalFlip(p=0.5)
    shift_scale_rotate = ShiftScaleRotate(p=0.8, scale_limit=0.2, rotate_limit=90)

    df = pd.read_csv(SAMPLE_SUBMISSION)

    x_test = load_data(data_path=TEST_DATA_2048)
    test_generator = ImageDataGenerator(x=x_test,
                                        batch_size=8,
                                        shuffle=True,
                                        horizontal_flip=horizontal_flip,
                                        shift_scale_rotate=shift_scale_rotate,
                                        input_shape=(2048, 2048, 3))

    indexes = test_generator.get_indexes()

    img = test_generator[0] * 255

    for i in range(img.shape[0]):
        index = indexes[i]
        prefix = df.iloc[index][0]
        r_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_red.tif".format(prefix)))
        g_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_green.tif".format(prefix)))
        b_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_blue.tif".format(prefix)))

        raw_data = np.stack([r_img, g_img, b_img], axis=-1)

        im_raw = Image.fromarray(np.uint8(raw_data))
        im_aug = Image.fromarray(np.uint8(img[i]))

        im_raw.save("{}_raw.png".format(prefix))
        im_aug.save("{}_from_generator.png".format(prefix))

if args.train:
    horizontal_flip = HorizontalFlip(p=0.5)
    shift_scale_rotate = ShiftScaleRotate(p=0.8, scale_limit=0.2, rotate_limit=90)

    df = pd.read_csv(TRAINING_DATA_CSV)

    x_test = load_data(data_path=TRAINING_DATA_2048)
    test_generator = ImageDataGenerator(x=x_test,
                                        batch_size=8,
                                        shuffle=True,
                                        horizontal_flip=horizontal_flip,
                                        shift_scale_rotate=shift_scale_rotate,
                                        input_shape=(2048, 2048, 3))

    indexes = test_generator.get_indexes()

    img = test_generator[0] * 255

    for i in range(img.shape[0]):
        index = indexes[i]
        prefix = df.iloc[index][0]
        r_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_red.tif".format(prefix)))
        g_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_green.tif".format(prefix)))
        b_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_blue.tif".format(prefix)))

        raw_data = np.stack([r_img, g_img, b_img], axis=-1)

        im_raw = Image.fromarray(np.uint8(raw_data))
        im_aug = Image.fromarray(np.uint8(img[i]))

        im_raw.save("{}_raw.png".format(prefix))
        im_aug.save("{}_from_generator.png".format(prefix))

if args.tta:
    df = pd.read_csv(TRAINING_DATA_CSV)

    img = load_data(data_path=TRAINING_DATA_2048)

    split_filename = os.path.join(DATA_DIR, "KFold_0.npz")
    split = np.load(file=split_filename)

    valid_indexes = split['test_indexes']

    valid_generators = get_test_time_augmentation_generators(image=img,
                                                             batch_size=8,
                                                             indexes=valid_indexes,
                                                             input_shape=(2048, 2048, 3))

    print(len(valid_generators[0]))
    print(len(valid_generators[0].get_indexes()))
    print(valid_generators[0][-1].shape)
    print(valid_generators[0][-2].shape)
    exit(0)

    prefix = df.iloc[2][0]
    r_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_red.tif".format(prefix)))
    g_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_green.tif".format(prefix)))
    b_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_blue.tif".format(prefix)))

    raw_data = np.stack([r_img, g_img, b_img], axis=-1)
    im_raw = Image.fromarray(np.uint8(raw_data))
    im_raw.save("{}_raw.png".format(prefix))

    for index, valid_generator in enumerate(valid_generators):
        img = valid_generator[0] * 255
        im_aug = Image.fromarray(np.uint8(img[2]))
        im_aug.save("{}_from_generator_{}.png".format(prefix, index))

