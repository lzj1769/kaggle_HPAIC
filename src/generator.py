from __future__ import print_function
from __future__ import division

import numpy as np
import math
from keras.utils import Sequence
import keras.backend as K


def image_augment(aug, image):
    if image.shape[2] == 4:
        image[:, :, 0:3] = aug(image=image[:, :, 0:3])['image']
        image[:, :, 3] = aug(image=image[:, :, 1:4])['image'][:, :, 2]
        return image

    else:
        image = aug(image=image)['image']
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
                 shuffle=False,
                 indexes=None,
                 input_shape=None,
                 learning_phase=False,
                 **kwargs):

        super(ImageDataGenerator, self).__init__()

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.learning_phase = learning_phase
        self.kwargs = kwargs

        if indexes is not None:
            self.indexes = indexes
        else:
            self.indexes = np.arange(self.x.shape[0])

        self.input_shape = input_shape
        self.n_samples = len(self.indexes)

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        if self.learning_phase:
            return int(math.floor(self.n_samples / self.batch_size))
            # return int(math.floor(self.n_samples / self.batch_size) / 2)
        else:
            return int(math.ceil(self.n_samples / self.batch_size))

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
        batch_x = np.empty(shape=(len(indexes), self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                           dtype=K.floatx())

        for i, index in enumerate(indexes):
            img = np.array(self.x[index], copy=True)

            # drop the yellow channel if using 3 channels
            if self.input_shape[2] == 3:
                img = img[:, :, :3]

            if self.kwargs:
                for key, aug in self.kwargs.items():
                    img = image_augment(aug, img)

            batch_x[i] = img

        batch_x /= 255.

        return batch_x

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_indexes(self):
        return self.indexes

    def get_n_samples(self):
        return self.n_samples


class BatchBalanceImageDataGenerator(Sequence):
    """
    Generate batches of images as well as their labels on the fly

    Parameters
    -----------
    """

    def __init__(self,
                 x=None,
                 y=None,
                 batch_size=16,
                 shuffle=False,
                 indexes=None,
                 input_shape=None,
                 **kwargs):

        super(BatchBalanceImageDataGenerator, self).__init__()

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.kwargs = kwargs
        self.indexes = indexes
        self.input_shape = input_shape
        self.indexes_0 = None
        self.indexes_1 = None

        self.on_epoch_end()

    def __len__(self):
        return int(math.floor(len(self.indexes_0) / self.batch_size) * 2)

    def __getitem__(self, index):
        # Generate indexes of the batch
        batch_size_0 = 12
        batch_size_1 = self.batch_size - batch_size_0

        indexes_0 = self.indexes_0[index * batch_size_0: (index + 1) * batch_size_0]
        indexes_1 = self.indexes_1[index * batch_size_1: (index + 1) * batch_size_1]

        indexes = np.concatenate((indexes_0, indexes_1))

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
        batch_x = np.empty(shape=(len(indexes), self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                           dtype=K.floatx())

        for i, index in enumerate(indexes):
            img = np.array(self.x[index], copy=True)

            # drop the yellow channel if using 3 channels
            if self.input_shape[2] == 3:
                img = img[:, :, :3]

            if self.kwargs:
                for key, aug in self.kwargs.items():
                    img = image_augment(aug, img)

            batch_x[i] = img

        batch_x /= 255.

        return batch_x

    def on_epoch_end(self):
        self.indexes_0 = np.intersect1d(np.argwhere(self.y == 0), self.indexes)
        np.random.shuffle(self.indexes_0)

        indexes_1 = np.intersect1d(np.argwhere(self.y == 1), self.indexes)
        np.random.shuffle(indexes_1)
        reps = len(self.indexes_0) // len(indexes_1) + 1
        self.indexes_1 = np.tile(indexes_1, reps)[:len(self.indexes_0)]


class UpSamplingImageDataGenerator(Sequence):
    """
    Generate batches of images as well as their labels on the fly

    Parameters
    -----------
    """

    def __init__(self,
                 x=None,
                 y=None,
                 batch_size=16,
                 shuffle=False,
                 indexes=None,
                 input_shape=None,
                 up_sampling_factor=None,
                 **kwargs):

        super(UpSamplingImageDataGenerator, self).__init__()

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.kwargs = kwargs
        self.input_shape = input_shape
        self.indexes_0 = np.intersect1d(np.argwhere(self.y == 0), indexes)
        self.indexes_1 = np.intersect1d(np.argwhere(self.y == 1), indexes)
        self.up_sampling_factor = up_sampling_factor
        self.indexes = np.concatenate((self.indexes_0, np.tile(self.indexes_1, self.up_sampling_factor)))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(math.floor(len(self.indexes) / self.batch_size))

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
        batch_x = np.empty(shape=(len(indexes), self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                           dtype=K.floatx())

        for i, index in enumerate(indexes):
            img = np.array(self.x[index], copy=True)

            # drop the yellow channel if using 3 channels
            if self.input_shape[2] == 3:
                img = img[:, :, :3]

            if self.kwargs:
                for key, aug in self.kwargs.items():
                    img = image_augment(aug, img)

            batch_x[i] = img

        batch_x /= 255.

        return batch_x

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)