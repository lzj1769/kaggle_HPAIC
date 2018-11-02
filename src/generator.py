from __future__ import print_function
from __future__ import division

import numpy as np
import scipy
import math
import cv2
import keras

from albumentations import HorizontalFlip, ShiftScaleRotate, RandomRotate90
from albumentations import RandomBrightness, Flip

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)

from keras.utils import Sequence
import keras.backend as K

from configure import *


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


class ImageDataGenerator(Sequence):
    """
    Generate batches of images as well as their labels on the fly

    Parameters
    -----------
    rotation_range: Int. Degree range for random rotations.

    width_shift_range: Float, 1-D array-like or int
        - float: fraction of total width, if < 1, or pixels if >= 1.
        - 1-D array-like: random elements from the array.
        - int: integer number of pixels from interval
            `(-width_shift_range, +width_shift_range)`
        - With `width_shift_range=2` possible values
            are integers `[-1, 0, +1]`,
            same as with `width_shift_range=[-1, 0, +1]`,
            while with `width_shift_range=1.0` possible values are floats
            in the interval [-1.0, +1.0).

    height_shift_range: Float, 1-D array-like or int
        - float: fraction of total height, if < 1, or pixels if >= 1.
        - 1-D array-like: random elements from the array.
        - int: integer number of pixels from interval
            `(-height_shift_range, +height_shift_range)`
        - With `height_shift_range=2` possible values
            are integers `[-1, 0, +1]`,
            same as with `height_shift_range=[-1, 0, +1]`,
            while with `height_shift_range=1.0` possible values are floats
            in the interval [-1.0, +1.0).

    shear_range: Float. Shear Intensity
        (Shear angle in counter-clockwise direction in degrees)

    zoom_range: Float or [lower, upper]. Range for random zoom.
        If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.

    fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
        Default is 'nearest'.
        Points outside the boundaries of the input are filled
        according to the given mode:
        - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
        - 'nearest':  aaaaaaaa|abcd|dddddddd
        - 'reflect':  abcddcba|abcd|dcbaabcd
        - 'wrap':  abcdabcd|abcd|abcdabcd

    cval: Float or Int.
        Value used for points outside the boundaries
        when `fill_mode = "constant"`.

    horizontal_flip: Boolean. Randomly flip inputs horizontally.

    vertical_flip: Boolean. Randomly flip inputs vertically.

    rescale: rescaling factor. Defaults to None.
        If None or 0, no rescaling is applied,
        otherwise we multiply the data by the value provided
        (after applying all other transformations).

    data_format: Image data format,
        either "channels_first" or "channels_last".
        "channels_last" mode means that the images should have shape
        `(samples, height, width, channels)`,
        "channels_first" mode means that the images should have shape
        `(samples, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".
    """

    def __init__(self, x, y=None, batch_size=16, n_classes=N_LABELS,
                 shuffle=False, augment=False, rescale=1.0 / 255,
                 output_shape=(IMAGE_HEIGHT, IMAGE_WIDTH), n_channels=N_CHANNELS,
                 rotation_range=0, width_shift_range=0., height_shift_range=0.,
                 shear_range=0., zoom_range=0., fill_mode='nearest', cval=0.,
                 horizontal_flip=False, vertical_flip=False,
                 data_format='channels_last',
                 preprocessing_function=None,
                 augment_prob=0.9):
        super(ImageDataGenerator, self).__init__()

        self.x = None
        self.y = y
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.rescale = rescale
        self.indexes = np.arange(x.shape[0])

        if self.shuffle:
            np.random.shuffle(self.indexes)

        self.output_shape = output_shape
        self.n_channels = n_channels

        self.data_format = data_format
        if self.data_format == "channels_last":
            self.row_axis = 1
            self.col_axis = 2
            self.channel_axis = 3
        else:
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3

        # parameters for image augmentation
        self.rotation_range = rotation_range
        self.height_shift_range = height_shift_range
        self.width_shift_range = width_shift_range
        self.shear_range = shear_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.fill_mode = fill_mode
        self.cval = cval

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received: %s' % (zoom_range,))

        self.pre_processing(x)
        self.preprocessing_function = preprocessing_function
        self.augment_prob = augment_prob

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        batch_x = self.__data_generation(indexes)

        if self.preprocessing_function:
            for i in range(batch_x.shape[0]):
                batch_x[i] = self.preprocessing_function(batch_x[i],
                                                         backend=keras.backend,
                                                         layers=keras.layers,
                                                         models=keras.models,
                                                         utils=keras.utils)
        else:
            for i in range(batch_x.shape[0]):
                batch_x[i] = batch_x[i] * self.rescale

        if self.y is not None:
            batch_y = self.y[indexes].astype(K.floatx())
            return batch_x, batch_y

        else:
            return batch_x

    def __data_generation(self, indexes):
        """
        Generates data containing batch_size samples' # X : (n_samples, IMAGE_HEIGHT, IMAGE_WIDTH, n_channels)
        """
        batch_x = np.empty((len(indexes), self.output_shape[0], self.output_shape[1], self.n_channels),
                           dtype=K.floatx())

        if self.augment:
            for i, idx in enumerate(indexes):
                if np.random.rand() < self.augment_prob:
                    batch_x[i] = self.random_transform(self.x[idx].astype(K.floatx()))
                else:
                    batch_x[i] = self.x[idx].astype(K.floatx())
        else:
            for i, idx in enumerate(indexes):
                batch_x[i] = self.x[idx].astype(K.floatx())

        return batch_x

    def pre_processing(self, x):
        """Pre-processing the image
        """
        # drop the yellow channel if using 3 channels
        if self.n_channels == 3:
            x = x[:, :, :, :3]

        # resize images if the model requires different input shape
        if self.output_shape != (x.shape[self.row_axis], x.shape[self.col_axis]):
            self.x = np.empty((x.shape[0], self.output_shape[0], self.output_shape[1], self.n_channels), dtype=np.uint8)
            for i in range(x.shape[0]):
                self.x[i] = cv2.resize(x[i], (self.output_shape[0], self.output_shape[1]))

        else:
            self.x = x

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.x.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_indexes(self):
        return self.indexes

    def random_transform(self, x, seed=None):
        """Applies a random transformation to an image.

        # Arguments
            x: 3D tensor, single image.
            seed: Random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """

        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        if scipy is None:
            raise ImportError('Image transformations require SciPy. '
                              'Install SciPy.')

        # rotate the image
        transform_matrix = None
        if self.rotation_range:
            theta = np.random.uniform(-self.rotation_range, self.rotation_range)
            theta = np.deg2rad(theta)

            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])

            transform_matrix = rotation_matrix

        # shift the image
        if self.height_shift_range:
            try:  # 1-D array-like or int
                tx = np.random.choice(self.height_shift_range)
                tx *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                tx = np.random.uniform(-self.height_shift_range,
                                       self.height_shift_range)
            if np.max(self.height_shift_range) < 1:
                tx *= x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            try:  # 1-D array-like or int
                ty = np.random.choice(self.width_shift_range)
                ty *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                ty = np.random.uniform(-self.width_shift_range,
                                       self.width_shift_range)
            if np.max(self.width_shift_range) < 1:
                ty *= x.shape[img_col_axis]
        else:
            ty = 0

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = shift_matrix
            else:
                transform_matrix = np.dot(transform_matrix, shift_matrix)

        # Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
            shear = np.deg2rad(shear)
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = shear_matrix
            else:
                transform_matrix = np.dot(transform_matrix, shear_matrix)

        # Range for random zoom
        if self.zoom_range[0] != 1 or self.zoom_range[1] != 1:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = zoom_matrix
            else:
                transform_matrix = np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = np.rollaxis(x, img_channel_axis, 0)
            final_affine_matrix = transform_matrix[:2, :2]
            final_offset = transform_matrix[:2, 2]

            channel_images = [scipy.ndimage.interpolation.affine_transform(
                x_channel,
                final_affine_matrix,
                final_offset,
                order=1,
                mode=self.fill_mode,
                cval=self.cval) for x_channel in x]
            x = np.stack(channel_images, axis=0)
            x = np.rollaxis(x, 0, img_channel_axis + 1)

        if (np.random.random() < 0.5) * self.horizontal_flip:
            x = flip_axis(x, img_col_axis)

        if (np.random.random() < 0.5) * self.vertical_flip:
            x = flip_axis(x, img_row_axis)

        return x
