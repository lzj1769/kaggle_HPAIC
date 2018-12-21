from __future__ import print_function, division
import os
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from albumentations import HorizontalFlip
from albumentations.augmentations import functional as F
from albumentations import DualTransform

from configure import *


def load_data(data_path=None):
    return np.load(data_path, mmap_mode='r')


def get_data_path(input_shape=None):
    training_data_path = None
    test_data_path = None

    if input_shape[0] == 512:
        training_data_path = TRAINING_DATA_512
        test_data_path = TEST_DATA_512

    elif input_shape[0] == 1024:
        training_data_path = TRAINING_DATA_1024
        test_data_path = TEST_DATA_1024

    elif input_shape[0] == 2048:
        training_data_path = TRAINING_DATA_2048
        test_data_path = TEST_DATA_2048

    return training_data_path, test_data_path


def get_target():
    mlb = MultiLabelBinarizer(classes=range(N_LABELS))
    target = list()

    df = pd.read_csv(TRAINING_DATA_CSV)
    for i in range(df.shape[0]):
        target.append(map(int, df.iloc[i][1].split(" ")))

    df = pd.read_csv(HPAV18_CSV)
    for i in range(df.shape[0]):
        target.append(map(int, df.iloc[i][1].split(" ")))

    return mlb.fit_transform(target)


def get_ids():
    ids = list()
    df = pd.read_csv(TRAINING_DATA_CSV)
    for i in range(df.shape[0]):
        ids.append(df.iloc[i][0])

    df = pd.read_csv(HPAV18_CSV)
    for i in range(df.shape[0]):
        ids.append(df.iloc[i][0])

    return ids


def generate_exp_config(net_name, k_fold=None):
    if k_fold is not None:
        return "{}_KFold_{}".format(net_name, k_fold)
    else:
        return net_name


def get_history_path(net_name):
    return os.path.join(MODEL_HISTORY_PATH, net_name)


def get_weights_path(net_name):
    return os.path.join(MODEL_WEIGHTS_PATH, net_name)


def get_acc_loss_path(net_name):
    return os.path.join(MODEL_ACC_LOSS_PATH, net_name)


def get_training_predict_path(net_name):
    return os.path.join(TRAINING_OUTPUT_PATH, net_name)


def get_test_predict_path(net_name):
    return os.path.join(TEST_OUTPUT_PATH, net_name)


def get_submission_path(net_name):
    return os.path.join(SUBMISSION_PATH, net_name)


def f1_scores_threshold(y_true, y_prab, thresholds):
    f1_scores = []
    for threshold in thresholds:
        y_pred = y_prab > threshold
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        f1_scores.append(f1)

    return f1_scores


def calculating_fraction(y_pred, threshold):
    fraction = []

    for i in range(N_LABELS):
        prab = y_pred[:, i]
        frac = 1.0 * np.sum(prab >= threshold[i]) / y_pred.shape[0]

        fraction.append(frac)

    return fraction


def calculating_threshold(y_pred, fraction):
    threshod = []

    for i in range(N_LABELS):
        prab = y_pred[:, i]
        frac = fraction[i]

        threshod.append(np.quantile(prab, 1 - frac))

    return np.array(threshod)


def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    n_dim = y_true.shape[1]
    weights = np.empty([n_dim, 2])
    for i in range(n_dim):
        weights[i] = compute_class_weight('balanced', [0., 1.], y_true[:, i])

    return weights


class RandomRotate90(DualTransform):
    """Randomly rotate the input by 90 degrees zero or more times.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32
    """

    def __init__(self, factor, p=1.0):
        super(RandomRotate90, self).__init__(p)
        self.factor = factor

    def apply(self, img, **params):
        """
        Args:
            factor (int): number of times the input will be rotated by 90 degrees.
        """
        return np.ascontiguousarray(np.rot90(img, self.factor))

    def get_params(self):
        return {'factor': self.factor}

    def apply_to_bbox(self, bbox, factor=0, **params):
        return F.bbox_rot90(bbox, factor, **params)


def get_test_time_augmentation_generators(image, batch_size=None, indexes=None, input_shape=None):
    from generator import ImageDataGenerator

    horizontal_flip = HorizontalFlip(p=1.0)
    random_rotate_90_1 = RandomRotate90(factor=1)
    random_rotate_90_2 = RandomRotate90(factor=2)
    random_rotate_90_3 = RandomRotate90(factor=3)

    # using raw image
    generator_1 = ImageDataGenerator(x=image,
                                     batch_size=batch_size,
                                     indexes=indexes,
                                     input_shape=input_shape)

    # generator_2 = ImageDataGenerator(x=image,
    #                                  batch_size=batch_size,
    #                                  indexes=indexes,
    #                                  input_shape=input_shape,
    #                                  random_rotate_90_1=random_rotate_90_1)
    #
    # generator_3 = ImageDataGenerator(x=image,
    #                                  batch_size=batch_size,
    #                                  indexes=indexes,
    #                                  input_shape=input_shape,
    #                                  random_rotate_90_2=random_rotate_90_2)
    #
    # generator_4 = ImageDataGenerator(x=image,
    #                                  batch_size=batch_size,
    #                                  indexes=indexes,
    #                                  input_shape=input_shape,
    #                                  random_rotate_90_3=random_rotate_90_3)
    #
    # generator_5 = ImageDataGenerator(x=image,
    #                                  batch_size=batch_size,
    #                                  indexes=indexes,
    #                                  input_shape=input_shape,
    #                                  horizontal_flip=horizontal_flip)
    #
    # generator_6 = ImageDataGenerator(x=image,
    #                                  batch_size=batch_size,
    #                                  indexes=indexes,
    #                                  input_shape=input_shape,
    #                                  horizontal_flip=horizontal_flip,
    #                                  random_rotate_90_1=random_rotate_90_1)
    #
    # generator_7 = ImageDataGenerator(x=image,
    #                                  batch_size=batch_size,
    #                                  indexes=indexes,
    #                                  input_shape=input_shape,
    #                                  horizontal_flip=horizontal_flip,
    #                                  random_rotate_90_2=random_rotate_90_2)
    #
    # generator_8 = ImageDataGenerator(x=image,
    #                                  batch_size=batch_size,
    #                                  indexes=indexes,
    #                                  input_shape=input_shape,
    #                                  horizontal_flip=horizontal_flip,
    #                                  random_rotate_90_3=random_rotate_90_3)

    # return [generator_1, generator_2,
    #         generator_3, generator_4,
    #         generator_5, generator_6,
    #         generator_7, generator_8]

    return [generator_1]
