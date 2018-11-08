from __future__ import print_function

import io
import os
import csv
import six
import warnings

import numpy as np
import pandas as pd
import time

from collections import OrderedDict
from collections import Iterable
from keras.callbacks import Callback, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


class CSVPDFLogger(Callback):
    """Callback that streams epoch results to a csv file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Example

    ```python
    csv_logger = CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```

    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, plot_filename=None, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.plot_filename = plot_filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        if six.PY2:
            self.file_flags = 'b'
            self._open_args = {}
        else:
            self.file_flags = ''
            self._open_args = {'newline': '\n'}
        super(CSVPDFLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = io.open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ['epoch'] + self.keys
            if six.PY2:
                fieldnames = [unicode(x) for x in fieldnames]
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

        # plot the training loss and accuracy
        df = pd.read_csv(self.filename)
        plt.style.use("ggplot")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        ax1.plot(df['classification_binary_accuracy'])
        ax1.plot(df['val_classification_binary_accuracy'])
        ax1.set_title('Accuracy')
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'validation'], loc='upper left')

        ax2.plot(df['classification_loss'])
        ax2.plot(df['val_classification_loss'])
        ax2.set_title('Classification Loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['train', 'validation'], loc='upper left')

        ax3.plot(df['reconstruction_loss'])
        ax3.plot(df['val_reconstruction_loss'])
        ax3.set_title('Reconstruction Loss')
        ax3.set_ylabel('loss')
        ax3.set_xlabel('epoch')
        ax3.legend(['train', 'validation'], loc='upper left')

        fig.tight_layout()
        fig.savefig(self.plot_filename)

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None


def build_callbacks(weights_path, logs_path, acc_loss_path, exp_config):
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
                                  factor=0.1,
                                  patience=4,
                                  min_lr=1e-06,
                                  min_delta=0.,
                                  verbose=1)

    filename = os.path.join(logs_path, "{}.log".format(exp_config))
    plot_filename = os.path.join(acc_loss_path, "{}.pdf".format(exp_config))
    csv_pdf_logger = CSVPDFLogger(filename=filename, plot_filename=plot_filename, append=True)

    callbacks = [check_pointer, early_stopper, reduce_lr, csv_pdf_logger]

    return callbacks
