from __future__ import print_function, division

import os
import sys
import numpy as np
import pandas as pd

from skmultilearn.model_selection import IterativeStratification
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

df = pd.read_csv(TRAINING_LABELS)

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

stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[TEST_SIZE, 1.0-TEST_SIZE])
train_indexes, test_indexes = next(stratifier.split(img, labels))

X_train, y_train = img[train_indexes, :], labels[train_indexes, :]
X_test, y_test = img[test_indexes, :], labels[test_indexes, :]

np.savez(TRAINING_DATA, img=X_train, label=y_train)
np.savez(VALIDATION_DATA, img=X_test, label=y_test)

df_train = df.iloc[train_indexes]
df_validation = df.iloc[test_indexes]

df_train.to_csv(TRAINING_DATA_CSV, index=False)
df_validation.to_csv(VALIDATION_DATA_CSV, index=False)
