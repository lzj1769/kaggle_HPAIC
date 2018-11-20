from __future__ import print_function, division

import os
import sys
import numpy as np
import pandas as pd
import cv2

from PIL import Image
from configure import *

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

print("processing test data...", file=sys.stderr)

df = pd.read_csv(SAMPLE_SUBMISSION)
test_img = np.empty((df.shape[0], IMAGE_WIDTH_1024, IMAGE_HEIGHT_1024, N_CHANNELS), dtype=np.uint8)

for i in range(df.shape[0]):
    if i % 500 == 0:
        print("processing {} images".format(i), file=sys.stderr)

    prefix = df.iloc[i][0]
    r_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_red.tif".format(prefix)))
    g_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_green.tif".format(prefix)))
    b_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_blue.tif".format(prefix)))
    y_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_yellow.tif".format(prefix)))

    img = np.stack([r_img, g_img, b_img, y_img], axis=-1)
    if (img.shape[0], img.shape[1]) != (IMAGE_WIDTH_1024, IMAGE_HEIGHT_1024):
        img = cv2.resize(img, (IMAGE_WIDTH_1024, IMAGE_HEIGHT_1024))

    test_img[i] = img

np.save(TEST_DATA_1024, test_img)

del test_img

df = pd.read_csv(TRAINING_DATA_CSV)
train_img = np.empty((df.shape[0], IMAGE_WIDTH_1024, IMAGE_HEIGHT_1024, N_CHANNELS), dtype=np.uint8)

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

np.save(TRAINING_DATA_1024, train_img)
