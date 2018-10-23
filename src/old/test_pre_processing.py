from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
import argparse

from PIL import Image
from configure import *

OUTPUT_DIR = "/home/rs619065/kaggle_HPAIC/src/test"

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true")
parser.add_argument("--train", default=False, action="store_true")
parser.add_argument("--validation", default=False, action="store_true")
parser.add_argument("--n_img", type=int, default=3)
args = parser.parse_args()

if args.test:
    print("processing test data...", file=sys.stderr)

    df = pd.read_csv(SAMPLE_SUBMISSION)

    test_data = np.load(TEST_DATA)

    test_img = test_data['img']

    random_samples = np.random.choice(test_img.shape[0], args.n_img)

    for sample in random_samples:
        prefix = df.iloc[sample][0]
        r_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_red.png".format(prefix)))
        g_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_green.png".format(prefix)))
        b_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_blue.png".format(prefix)))
        y_img = Image.open(os.path.join(TEST_INPUT_DIR, "{}_yellow.png".format(prefix)))

        raw_data = np.stack([r_img, g_img, b_img, y_img], axis=-1)
        test_data = test_img[sample]

        im_raw = Image.fromarray(np.uint8(raw_data))
        im_test = Image.fromarray(np.uint8(test_data))

        im_raw.save(fp=os.path.join(OUTPUT_DIR, "{}_raw.png".format(prefix)))
        im_test.save(fp=os.path.join(OUTPUT_DIR, "{}_test.png".format(prefix)))

if args.train:
    print("processing training data...", file=sys.stderr)
    df = pd.read_csv(TRAINING_DATA_CSV)

    train_data = np.load(TRAINING_DATA)

    train_img = train_data['img']