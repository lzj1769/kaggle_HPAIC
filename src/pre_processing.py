from __future__ import print_function, division

import os
import sys
import numpy as np
import pandas as pd

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
df_validation = df.sample(frac=TEST_SIZE, replace=False, random_state=42)
df_train = df.drop(df_validation.index)

print("processing training data...", file=sys.stderr)

training_img = np.empty((df_train.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 4),
                        dtype=np.uint8)
training_label = list()

for i in range(df_train.shape[0]):
    prefix = df_train.iloc[i][0]
    r_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_red.png".format(prefix)))
    g_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_green.png".format(prefix)))
    b_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_blue.png".format(prefix)))
    y_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_yellow.png".format(prefix)))

    training_img[i] = np.stack([r_img, g_img, b_img, y_img], axis=-1)

np.savez(TRAINING_DATA, img=training_img)
del training_img

print("processing validation data...", file=sys.stderr)

validation_img = np.empty((df_validation.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 4),
                          dtype=np.uint8)

for i in range(df_validation.shape[0]):
    prefix = df_validation.iloc[i][0]
    r_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_red.png".format(prefix)))
    g_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_green.png".format(prefix)))
    b_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_blue.png".format(prefix)))
    y_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_yellow.png".format(prefix)))

    validation_img[i] = np.stack([r_img, g_img, b_img, y_img], axis=-1)

np.savez(VALIDATION_DATA, img=validation_img)

df_train.to_csv(TRAINING_DATA_CSV, index=False)
df_validation.to_csv(VALIDATION_DATA_CSV, index=False)
