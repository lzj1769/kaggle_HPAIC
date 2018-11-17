from __future__ import print_function
from __future__ import division

import numpy as np
import math
import cv2
from keras.utils import Sequence
import keras.backend as K


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
            return math.floor(self.n_samples / self.batch_size)
        else:
            return math.ceil(self.n_samples / self.batch_size)

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        batch_x = self.generate_data(indexes)

        if self.y is not None:
            batch_y = np.empty(shape=(self.batch_size, self.y.shape[1]), dtype=K.floatx())

            for i, index in enumerate(indexes):
                batch_y[i] = self.y[index]

            return batch_x, batch_y

        else:
            return batch_x

    def generate_data(self, indexes):
        """
        Generates data containing batch_size samples' # X : (n_samples, IMAGE_HEIGHT, IMAGE_WIDTH, n_channels)
        """
        batch_x = np.empty(shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                           dtype=K.floatx())

        for i, index in enumerate(indexes):
            img = self.x[index]

            if self.kwargs:
                for key, aug in self.kwargs.items():
                    img = image_augment(aug, img)

            # drop the yellow channel if using 3 channels
            if self.input_shape[2] == 3:
                img = img[:, :, :3]

            if (img.shape[0], img.shape[1]) != (self.input_shape[0], self.input_shape[1]):
                img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))

            batch_x[i] = img

        batch_x /= 255.

        return batch_x

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

# class ImageDataGeneratorFromDirectory(Sequence):
#     """
#     Generate batches of images as well as their labels on the fly
#
#     Parameters
#     -----------
#     """
#
#     def __init__(self,
#                  image_path=None,
#                  target=None,
#                  batch_size=16,
#                  shuffle=False,
#                  input_shape=None,
#                  learning_phase=False,
#                  **kwargs):
#
#         super(ImageDataGeneratorFromDirectory, self).__init__()
#
#         self.image_path = image_path
#         self.target = target
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.indexes = np.arange(len(image_path))
#
#         self.learning_phase = learning_phase
#         self.kwargs = kwargs
#         self.input_shape = input_shape
#
#         if self.shuffle:
#             np.random.shuffle(self.indexes)
#
#     def __len__(self):
#         if self.learning_phase:
#             return math.floor(len(self.indexes) / self.batch_size)
#         else:
#             return math.ceil(len(self.indexes) / self.batch_size)
#
#     def __getitem__(self, index):
#         # Generate indexes of the batch
#         indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
#
#         batch_x = self.generate_data(indexes)
#         if self.target is not None:
#             batch_y = self.target[indexes]
#             return batch_x, batch_y
#         else:
#             return batch_x
#
#     def generate_data(self, indexes):
#         """
#         Generates data containing batch_size samples' # X : (n_samples, IMAGE_HEIGHT, IMAGE_WIDTH, n_channels)
#         """
#         batch_x = np.empty((self.batch_size, self.input_shape[0],
#                             self.input_shape[1], self.input_shape[2]),
#                            dtype=K.floatx())
#
#         for i, index in enumerate(indexes):
#             r_img = Image.open(self.image_path[index] + "_red.tif")
#             g_img = Image.open(self.image_path[index] + "_green.tif")
#             b_img = Image.open(self.image_path[index] + "_blue.tif")
#
#             # drop the yellow channel
#             if self.input_shape[2] == 3:
#                 img = np.stack([r_img, g_img, b_img], axis=-1)
#
#             else:
#                 y_img = Image.open(self.image_path[index] + "_yellow.tif")
#                 img = np.stack([r_img, g_img, b_img, y_img], axis=-1)
#
#             # resize the image if needed
#             if (img.shape[0], img.shape[1]) != (self.input_shape[0], self.input_shape[1]):
#                 img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
#
#             # image augmentation
#             if self.kwargs:
#                 for key, aug in self.kwargs.items():
#                     img = image_augment(aug, img)
#
#             batch_x[i] = img
#
#         batch_x /= 255.
#
#         return batch_x
#
#     def on_epoch_end(self):
#         # Updates indexes after each epoch
#         if self.shuffle:
#             np.random.shuffle(self.indexes)
