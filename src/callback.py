from __future__ import print_function

import os
import warnings

import numpy as np

from keras.callbacks import Callback, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger

import time


class EarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 seconds=None,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.seconds = seconds
        self.start_time = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.start_time = time.time()
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        # check the monitor quantity
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

        # check the running time
        if time.time() - self.start_time > self.seconds:
            self.model.stop_training = True
            if self.verbose > 0:
                print('Stopping after %s seconds.' % self.seconds)
            if self.restore_best_weights:
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value


# class F1ScoreChecker(Callback):
#     """Calculate F1 score after training for each epoch.
#
#     # Arguments
#         verbose: verbosity mode.
#     """
#
#     def __init__(self, x_val, y_val, fraction):
#         super(F1ScoreChecker, self).__init__()
#         self.x_val = x_val
#         self.y_val = y_val
#         self.fraction = fraction
#         self.n_labels = len(fraction)
#         self.val_f1 = None
#
#     def on_train_begin(self, logs=None):
#         self.val_f1 = []
#
#     def on_epoch_end(self, epoch, logs=None):
#         y_pred = self.model.predict_generator(self.x_val, use_multiprocessing=True, workers=8)
#         indexes = self.x_val.get_indexes()
#         y_val = self.y_val[indexes]
#
#         threshod = []
#         for i, frac in enumerate(self.fraction):
#             prab = y_pred[:, i]
#             threshod.append(np.quantile(prab, 1 - frac))
#
#         threshod = np.array(threshod)
#
#         valid_labels = list()
#         # convert the predicted probabilities into labels for validation data
#         for i in range(y_pred.shape[0]):
#             label_predict = np.arange(self.n_labels)[np.greater(y_pred[i], threshod)]
#             if label_predict.size == 0:
#                 label_predict = [np.argmax(y_pred[i])]
#
#             valid_labels.append(label_predict)
#
#         mlb = MultiLabelBinarizer(classes=range(self.n_labels))
#         valid_labels_bin = mlb.fit_transform(valid_labels)
#
#         f1_score_val = f1_score(y_true=y_val, y_pred=valid_labels_bin, average="macro").round(3)
#
#         self.val_f1.append(f1_score_val)


def build_callbacks(weights_path, logs_path, exp_config):
    fp = os.path.join(weights_path, "{}.h5".format(exp_config))
    check_pointer = ModelCheckpoint(filepath=fp,
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True)

    early_stopper = EarlyStopping(monitor='val_loss',
                                  patience=20,
                                  seconds=3600 * 7,
                                  verbose=1,
                                  restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=5,
                                  min_lr=1e-08,
                                  min_delta=0.,
                                  verbose=1)

    filename = os.path.join(logs_path, "{}.log".format(exp_config))
    csv_logger = CSVLogger(filename=filename, append=True)

    callbacks = [check_pointer, early_stopper, reduce_lr, csv_logger]

    return callbacks
