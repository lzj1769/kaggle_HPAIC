from __future__ import print_function, division

import os
import sys
import numpy as np
import pandas as pd
import h5py
import cv2

from sklearn.preprocessing import MultiLabelBinarizer

from PIL import Image
from configure import *

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

print("processing test data...", file=sys.stderr)

df = pd.read_csv(SAMPLE_SUBMISSION)
test_img = np.empty((df.shape[0], IMAGE_WIDTH_FULL, IMAGE_HEIGHT_FULL, N_CHANNELS),
                    dtype=np.uint8)

f = h5py.File(FULL_SIZE_TEST_DATA, "w")
f.create_dataset(name="img", shape=(df.shape[0], IMAGE_WIDTH_FULL, IMAGE_HEIGHT_FULL, N_CHANNELS),
                 dtype=np.uint8)

for i in range(df.shape[0]):
    prefix = df.iloc[i][0]
    r_img = Image.open(os.path.join(FULL_SIZE_TEST_INPUT_DIR, "{}_red.tif".format(prefix)))
    g_img = Image.open(os.path.join(FULL_SIZE_TEST_INPUT_DIR, "{}_green.tif".format(prefix)))
    b_img = Image.open(os.path.join(FULL_SIZE_TEST_INPUT_DIR, "{}_blue.tif".format(prefix)))
    y_img = Image.open(os.path.join(FULL_SIZE_TEST_INPUT_DIR, "{}_yellow.tif".format(prefix)))

    img = np.stack([r_img, g_img, b_img, y_img], axis=-1)
    if (img.shape[0], img.shape[1]) != (IMAGE_WIDTH_FULL, IMAGE_HEIGHT_FULL):
        img = cv2.resize(img, (IMAGE_WIDTH_FULL, IMAGE_HEIGHT_FULL))

    f["img"][i, ...] = img

f.close()

# df = pd.read_csv(TRAINING_DATA_CSV)
#
# f = h5py.File(FULL_SIZE_TRAINING_DATA, "w")
# f.create_dataset(name="img", shape=(df.shape[0], IMAGE_WIDTH_FULL, IMAGE_HEIGHT_FULL, N_CHANNELS),
#                  dtype=np.uint8)
# f.create_dataset(name="label", shape=(df.shape[0], N_LABELS), dtype=np.uint8)
#
# mlb = MultiLabelBinarizer(classes=range(N_LABELS))
#
# for i in range(df.shape[0]):
#     prefix = df.iloc[i][0]
#     r_img = Image.open(os.path.join(FULL_SIZE_TRAINING_INPUT_DIR, "{}_red.tif".format(prefix)))
#     g_img = Image.open(os.path.join(FULL_SIZE_TRAINING_INPUT_DIR, "{}_green.tif".format(prefix)))
#     b_img = Image.open(os.path.join(FULL_SIZE_TRAINING_INPUT_DIR, "{}_blue.tif".format(prefix)))
#     y_img = Image.open(os.path.join(FULL_SIZE_TRAINING_INPUT_DIR, "{}_yellow.tif".format(prefix)))
#
#     img = np.stack([r_img, g_img, b_img, y_img], axis=-1)
#     if (img.shape[0], img.shape[1]) != (IMAGE_WIDTH_FULL, IMAGE_HEIGHT_FULL):
#         img = cv2.resize(img, (IMAGE_WIDTH_FULL, IMAGE_HEIGHT_FULL))
#
#     f["img"][i, ...] = img
#     f["label"][i, ...] = mlb.fit_transform(map(int, df.iloc[i][1].split(" ")))
#
# f.close()
