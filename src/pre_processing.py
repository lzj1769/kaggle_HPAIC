from __future__ import print_function, division

import os
import sys
import numpy as np
import pandas as pd
import cv2

from PIL import Image
from configure import *

# print("processing test data...", file=sys.stderr)
#
# df = pd.read_csv(SAMPLE_SUBMISSION)
# test_img = np.empty((df.shape[0], IMAGE_WIDTH_512, IMAGE_HEIGHT_512, N_CHANNELS), dtype=np.uint8)
#
# for i in range(df.shape[0]):
#     if i % 500 == 0:
#         print("processing {} images".format(i), file=sys.stderr)
#
#     prefix = df.iloc[i][0]
#     r_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_red.tif".format(prefix)))
#     g_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_green.tif".format(prefix)))
#     b_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_blue.tif".format(prefix)))
#     y_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_yellow.tif".format(prefix)))
#
#     img = np.stack([r_img, g_img, b_img, y_img], axis=-1)
#     if (img.shape[0], img.shape[1]) != (IMAGE_WIDTH_512, IMAGE_HEIGHT_512):
#         img = cv2.resize(img, (IMAGE_WIDTH_512, IMAGE_HEIGHT_512))
#
#     test_img[i] = img
#
# np.save(TEST_DATA_512, test_img)
#
# del test_img


df = pd.read_csv(TRAINING_DATA_CSV)
df_hpa_v18 = pd.read_csv(HPAV18_CSV)

train_img = np.empty((df.shape[0]+df_hpa_v18.shape[0], IMAGE_WIDTH_1024, IMAGE_HEIGHT_1024, N_CHANNELS), dtype=np.uint8)

for i in range(df.shape[0]):
    if i % 500 == 0:
        print("processing {} images".format(i), file=sys.stderr)

    prefix = df.iloc[i][0]
    r_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_red.tif".format(prefix)))
    g_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_green.tif".format(prefix)))
    b_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_blue.tif".format(prefix)))
    y_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_yellow.tif".format(prefix)))

    img = np.stack([r_img, g_img, b_img, y_img], axis=-1)
    if (img.shape[0], img.shape[1]) != (IMAGE_WIDTH_1024, IMAGE_HEIGHT_1024):
        img = cv2.resize(img, (IMAGE_WIDTH_1024, IMAGE_HEIGHT_1024))

    train_img[i] = img

for i in range(df_hpa_v18.shape[0]):
    if i % 500 == 0:
        print("processing {} images".format(i+df.shape[0]), file=sys.stderr)

    prefix = df_hpa_v18.iloc[i][0]
    r_img = np.array(Image.open(os.path.join(HPAV18_DIR), "{}_red.jpg".format(prefix)))
    g_img = np.array(Image.open(os.path.join(HPAV18_DIR), "{}_green.jpg".format(prefix)))
    b_img = np.array(Image.open(os.path.join(HPAV18_DIR), "{}_blue.jpg".format(prefix)))
    y_img = np.array(Image.open(os.path.join(HPAV18_DIR), "{}_yellow.jpg".format(prefix)))

    img = np.stack([r_img[:, :, 0], g_img[:, :, 1], b_img[:, :, 2], y_img[:, :, 0]], axis=-1)

    if (img.shape[0], img.shape[1]) != (IMAGE_WIDTH_1024, IMAGE_HEIGHT_1024):
        img = cv2.resize(img, (IMAGE_WIDTH_1024, IMAGE_HEIGHT_1024))

    train_img[i+df.shape[0]] = img


np.save(TRAINING_DATA_1024, train_img)


del train_img
