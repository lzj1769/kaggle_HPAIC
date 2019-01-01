from __future__ import print_function

import argparse
import importlib
import sys
import math

from keras.utils import Sequence
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.metrics import binary_accuracy
import keras.backend as K

from callback import build_callbacks
from generator import ImageDataGenerator
from sklearn.utils import class_weight
from utils import *
from configure import *

from albumentations import HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomBrightness


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--net_name", type=str, default="Siamese",
                        help='name of convolutional neural network.')
    parser.add_argument("-k", "--k_fold", type=int, default=0,
                        help="number of KFold split, should between 0 and 1")
    parser.add_argument("-e", "--epochs", type=int, default=100,
                        help="number of epochs for training. DEFAULT: 100")
    parser.add_argument("-g", "--n_gpus", type=int, default=2,
                        help="number of GPUS for training, DEFAULT: 2")
    parser.add_argument("-w", "--workers", type=int, default=8,
                        help="number of cores for training. DEFAULT: All 16 cpus")
    parser.add_argument("-v", "--verbose", type=int, default=2,
                        help="Verbosity mode. DEFAULT: 2")
    parser.add_argument("-l", "--label", type=int, default=None,
                        help="label of training")
    return parser.parse_args()


class PairedImageDataGenerator(Sequence):
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

        super(PairedImageDataGenerator, self).__init__()

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.kwargs = kwargs
        self.indexes = indexes
        self.input_shape = input_shape
        self.indexes_0 = np.intersect1d(np.argwhere(self.y == 0), self.indexes)
        self.indexes_1 = np.intersect1d(np.argwhere(self.y == 1), self.indexes)

    def __len__(self):
        return int(math.floor(len(self.indexes_0) / self.batch_size) * 2)

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes_0 = self.indexes_0[index * batch_size_0: (index + 1) * batch_size_0]
        indexes_1 = self.indexes_1[index * batch_size_1: (index + 1) * batch_size_1]

        indexes = np.concatenate((indexes_0, indexes_1))

        batch_x1, batch_x2 = self.generate_paired_images(indexes)

        if self.y is not None:
            batch_y = self.y[indexes].astype(K.floatx())
            return batch_x1, batch_x2, batch_y

        else:
            return batch_x1, batch_x2

    def generate_paired_images(self, indexes):
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

        np.random.shuffle(self.indexes_0)

        indexes_1 = np.intersect1d(np.argwhere(self.y == 1), self.indexes)
        np.random.shuffle(indexes_1)
        reps = len(self.indexes_0) // len(indexes_1) + 1
        self.indexes_1 = np.tile(indexes_1, reps)[:len(self.indexes_0)]


def main():
    args = parse_args()

    print("load the model configuration...", file=sys.stderr)
    print("===========================================================================\n", file=sys.stderr)

    exp_config = generate_exp_config_single_label(net_name=args.net_name, k_fold=args.k_fold, label=args.label)
    weights_path = get_weights_path(net_name=args.net_name)

    net = importlib.import_module("Nets." + args.net_name)

    batch_size = net.BATCH_SIZE
    input_shape = net.INPUT_SHAPE
    max_queue_size = net.MAX_QUEUE_SIZE
    learning_rate = net.LEARNING_RATE

    weights_filename = os.path.join(weights_path, "{}.h5".format(exp_config))
    model = net.build_model(num_classes=1)

    if os.path.exists(weights_filename):
        model.load_weights(weights_filename, by_name=True)
        optimizer = Adam(lr=learning_rate * 0.1)

    else:
        model.summary()
        optimizer = Adam(lr=learning_rate)

    # parallel_model = multi_gpu_model(model=model, gpus=args.n_gpus, cpu_merge=False)

    model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])

    # parallel_model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=[binary_accuracy])

    print("load training and validation data...", file=sys.stderr)
    print("===========================================================================\n", file=sys.stderr)

    train_data, _ = get_data_path(input_shape=input_shape)

    img = load_data(data_path=train_data)
    target = get_target()[:, args.label]

    split_filename = os.path.join(DATA_DIR, "Label_{}_KFold_{}.npz".format(args.label, args.k_fold))
    split = np.load(file=split_filename)

    train_indexes = split['train_indexes']
    test_indexes = split['test_indexes']

    n_train_pos = np.sum(target[train_indexes] == 1)
    n_train_neg = np.sum(target[train_indexes] == 0)
    n_valid_pos = np.sum(target[test_indexes] == 1)
    n_valid_neg = np.sum(target[test_indexes] == 0)
    print("Training model on {} positive and {} negative samples".format(n_train_pos, n_train_neg,
                                                                         file=sys.stderr))

    print("Validate model on {} positive and {} negative samples".format(n_valid_pos, n_valid_neg,
                                                                         file=sys.stderr))
    print("===========================================================================\n", file=sys.stderr)

    # set augmentation parameters
    horizontal_flip = HorizontalFlip(p=0.5)
    vertical_flip = VerticalFlip(p=0.5)
    shift_scale_rotate = ShiftScaleRotate(p=0.8, scale_limit=0.2, rotate_limit=90)
    random_brightness = RandomBrightness(p=0.1, limit=0.1)

    train_generator = ImageDataGenerator(x=img,
                                         y=target,
                                         indexes=train_indexes,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         input_shape=input_shape,
                                         horizontal_flip=horizontal_flip,
                                         vertical_flip=vertical_flip,
                                         shift_scale_rotate=shift_scale_rotate,
                                         random_brightness=random_brightness,
                                         learning_phase=True)

    valid_generator = ImageDataGenerator(x=img,
                                         y=target,
                                         indexes=test_indexes,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         input_shape=input_shape,
                                         learning_phase=True)

    history_path = get_history_path(net_name=args.net_name)
    acc_loss_path = get_acc_loss_path(net_name=args.net_name)
    callbacks = build_callbacks(model=model,
                                weights_path=weights_path,
                                history_path=history_path,
                                acc_loss_path=acc_loss_path,
                                exp_config=exp_config)

    print("training model...", file=sys.stderr)
    print("===========================================================================\n", file=sys.stderr)
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(target[train_indexes]),
                                                      target[train_indexes])

    print("class weights for negative {}, positive samples {}".format(class_weights[0], class_weights[1]))

    model.fit_generator(generator=train_generator,
                        validation_data=valid_generator,
                        epochs=args.epochs,
                        verbose=args.verbose,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        workers=args.workers,
                        max_queue_size=max_queue_size,
                        class_weight=class_weights)

    print("complete!!")
    K.clear_session()


if __name__ == '__main__':
    main()
