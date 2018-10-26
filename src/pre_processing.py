from __future__ import print_function, division

import os
import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

from PIL import Image
from configure import *

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

print("processing test data...", file=sys.stderr)

df = pd.read_csv(SAMPLE_SUBMISSION)
test_img = np.empty((df.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, 4), dtype=np.uint8)

for i in range(df.shape[0]):
    prefix = df.iloc[i][0]
    r_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_red.png".format(prefix)))
    g_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_green.png".format(prefix)))
    b_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_blue.png".format(prefix)))
    y_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_yellow.png".format(prefix)))

    test_img[i] = np.stack([r_img, g_img, b_img, y_img], axis=-1)

np.savez(TEST_DATA, img=test_img)

del test_img

df = pd.read_csv(TRAINING_DATA_CSV)

img = np.empty((df.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS), dtype=np.uint8)
labels = list()

mlb = MultiLabelBinarizer(classes=range(N_LABELS))

for i in range(df.shape[0]):
    prefix = df.iloc[i][0]
    r_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_red.png".format(prefix)))
    g_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_green.png".format(prefix)))
    b_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_blue.png".format(prefix)))
    y_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_yellow.png".format(prefix)))

    img[i] = np.stack([r_img, g_img, b_img, y_img], axis=-1)
    labels.append(map(int, df.iloc[i][1].split(" ")))

labels = mlb.fit_transform(labels)

np.savez(TRAINING_DATA, img=img, label=labels)
