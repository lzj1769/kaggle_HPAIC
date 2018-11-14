from __future__ import print_function
from __future__ import division

import numpy as np
import math
import cv2

from keras.utils import Sequence
import keras.backend as K

from configure import *


def image_augment(aug, image):
    image[:, :, 0:3] = aug(image=image[:, :, 0:3])['image']
    image[:, :, 3] = aug(image=image[:, :, 1:4])['image'][:, :, 2]

    return image


class ImageDataGenerator(Sequence):
    """
    Generate batches of images as well as their labels on the fly

    Parameters
    -----------
    """

    def __init__(self,
                 x=None, y=None,
                 batch_size=16,
                 n_classes=N_LABELS,
                 shuffle=False,
                 output_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
                 n_channels=N_CHANNELS,
                 learning_phase=False,
                 **kwargs):

        super(ImageDataGenerator, self).__init__()

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = np.arange(x.shape[0])
        self.learning_phase = learning_phase

        if self.shuffle:
            np.random.shuffle(self.indexes)

        self.output_shape = output_shape
        self.n_channels = n_channels

        self.kwargs = kwargs

    def __len__(self):
        if self.learning_phase:
            return math.floor(self.x.shape[0] / self.batch_size)
        else:
            return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        batch_x = self.generate_data(indexes)

        if self.y is not None:
            batch_y = self.y[indexes].astype(K.floatx())
            return batch_x, batch_y
        else:
            return batch_x

    def generate_data(self, indexes):
        """
        Generates data containing batch_size samples' # X : (n_samples, IMAGE_HEIGHT, IMAGE_WIDTH, n_channels)
        """
        batch_x = self.x[indexes].astype(K.floatx())

        # image augmentation
        if self.kwargs:
            for i in range(batch_x.shape[0]):
                for key, aug in self.kwargs.iteritems():
                    batch_x[i] = image_augment(aug, batch_x[i])

        # drop the yellow channel if using 3 channels
        if self.n_channels == 3:
            batch_x = batch_x[:, :, :, :3]

        # resize the image if need
        if self.output_shape != (batch_x.shape[1], batch_x.shape[2]):
            batch_x_resize = np.empty((batch_x.shape[0], self.output_shape[0], self.output_shape[1], self.n_channels),
                                      dtype=K.floatx())

            for i in range(batch_x.shape[0]):
                batch_x_resize[i] = cv2.resize(batch_x[i], (self.output_shape[0], self.output_shape[1]))

            batch_x = batch_x_resize

        batch_x /= 255.

        return batch_x

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.x.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_indexes(self):
        return self.indexes
