from __future__ import print_function, division

import numpy as np
import pandas as pd
import keras as K

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing
import time
import collections
import sys
import signal

from utils import *

# The original class can be imported like this:
# from keras.preprocessing.image import ImageDataGenerator

# We access the modified version through T.ImageDataGenerator
import generator2 as T

# Useful for checking the output of the generators after code change
try:
    from importlib import reload
    reload(T)
except:
    reload(T)


def preprocess_img(img):
    img = img.astype(np.float32) / 255.0
    img -= 0.5
    return img * 2


def plot_images(img_gen, title):
    fig, ax = plt.subplots(6, 6, figsize=(10, 10))
    plt.suptitle(title, size=32)
    plt.setp(ax, xticks=[], yticks=[])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    for (imgs, labels) in img_gen:
        for i in range(6):
            for j in range(6):
                if i*6 + j < 32:
                    ax[i][j].imshow(imgs[i*6 + j])
        break

    fig.savefig(os.path.join(TEST_PATH, "augument.pdf"))

# load data
print("load data...", file=sys.stderr)

train_img, train_label = load_data(dataset="train")
#try:
#    pool.terminate()
#except:
#    pass
n_process = 8

pool = multiprocessing.Pool(processes=n_process)
start = time.time()
gen = T.ImageDataGenerator(
     rotation_range=45,
     width_shift_range=.1,
     height_shift_range=.1,
     shear_range=0.,
     zoom_range=0,
     channel_shift_range=0,
     fill_mode='nearest',
     cval=0.,
     horizontal_flip=True,
     vertical_flip=False,
     rescale=1/255.,
     #preprocessing_function=preprocess_img, # disable for nicer visualization
     dim_ordering='default',
     pool=pool # <-------------- Only change needed!
)
X_train_aug = gen.flow(train_img, train_label, seed=0)

print('{} process, duration: {}'.format(n_process, time.time() - start))
plot_images(X_train_aug, 'Augmented Images generated with {} processes'.format(n_process))

pool.terminate()