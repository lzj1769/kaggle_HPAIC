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


def load_data(data_path=None, image_size=None):
    if image_size in [(IMAGE_WIDTH_512, IMAGE_HEIGHT_512), (IMAGE_WIDTH_1024, IMAGE_HEIGHT_2048)]:
        data = np.load(data_path)
    else:
        data = np.load(data_path, mmap_mode='r')

    return data


def get_target():
    df = pd.read_csv(TRAINING_DATA_CSV)
    mlb = MultiLabelBinarizer(classes=range(N_LABELS))
    target = list()
    for i in range(df.shape[0]):
        target.append(map(int, df.iloc[i][1].split(" ")))

    return mlb.fit_transform(target)


def generate_exp_config(net_name, k_fold=None):
    exp_config = net_name

    if k_fold is not None:
        return "{}_KFold_{}".format(exp_config, k_fold)
    else:
        return exp_config


def get_custom_objects(net_name):
    custom_objects = None
    if net_name == 'ResNet101':
        from Nets.ResNet101 import Scale
        custom_objects = {'Scale': Scale}

    elif net_name == 'ResNet152':
        from Nets.ResNet152 import Scale
        custom_objects = {'Scale': Scale}

    return custom_objects


def get_logs_path(net_name):
    return os.path.join(MODEL_LOG_PATH, net_name)


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

    def __init__(self, factor):
        super(RandomRotate90, self).__init__()
        self.factor = factor

    def apply(self, img, factor=0, **params):
        """
        Args:
            factor (int): number of times the input will be rotated by 90 degrees.
        """
        return np.ascontiguousarray(np.rot90(img, factor))

    def get_params(self):
        return {'factor': self.factor}

    def apply_to_bbox(self, bbox, factor=0, **params):
        return F.bbox_rot90(bbox, factor, **params)


def get_test_time_augmentation_generators(image, batch_size, output_shape, n_channels):
    from generator import ImageDataGenerator

    horizontal_flip = HorizontalFlip(p=1.0)
    random_rotate_90_1 = RandomRotate90(factor=1)
    random_rotate_90_2 = RandomRotate90(factor=2)
    random_rotate_90_3 = RandomRotate90(factor=3)

    # using raw image
    generator_1 = ImageDataGenerator(x=image,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     output_shape=output_shape,
                                     n_channels=n_channels)

    generator_2 = ImageDataGenerator(x=image,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     output_shape=output_shape,
                                     n_channels=n_channels,
                                     random_rotate_90_1=random_rotate_90_1)

    generator_3 = ImageDataGenerator(x=image,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     output_shape=output_shape,
                                     n_channels=n_channels,
                                     random_rotate_90_2=random_rotate_90_2)

    generator_4 = ImageDataGenerator(x=image,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     output_shape=output_shape,
                                     n_channels=n_channels,
                                     random_rotate_90_3=random_rotate_90_3)
    generator_5 = ImageDataGenerator(x=image,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     output_shape=output_shape,
                                     n_channels=n_channels,
                                     horizontal_flip=horizontal_flip)

    generator_6 = ImageDataGenerator(x=image,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     output_shape=output_shape,
                                     n_channels=n_channels,
                                     horizontal_flip=horizontal_flip,
                                     random_rotate_90_1=random_rotate_90_1)

    generator_7 = ImageDataGenerator(x=image,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     output_shape=output_shape,
                                     n_channels=n_channels,
                                     horizontal_flip=horizontal_flip,
                                     random_rotate_90_2=random_rotate_90_2)

    generator_8 = ImageDataGenerator(x=image,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     output_shape=output_shape,
                                     n_channels=n_channels,
                                     horizontal_flip=horizontal_flip,
                                     random_rotate_90_3=random_rotate_90_3)

    return [generator_1, generator_2,
            generator_3, generator_4,
            generator_5, generator_6,
            generator_7, generator_8]
