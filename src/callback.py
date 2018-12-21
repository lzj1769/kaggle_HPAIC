from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
import time
import warnings

from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.utils.io_utils import h5dict
from keras.engine.saving import _serialize_model

try:
    import h5py

    HDF5_OBJECT_HEADER_LIMIT = 64512
except ImportError:
    h5py = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class EarlyStoppingWithTime(EarlyStopping):
    """Stop training when a monitored quantity has stopped improving or the training
    time is achieved.

    """

    def __init__(self, seconds=None, **kwargs):
        self.seconds = seconds
        self.start_time = None
        super(EarlyStoppingWithTime, self).__init__(**kwargs)

    def on_train_begin(self, logs=None):
        super(EarlyStoppingWithTime, self).on_train_begin(logs)
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        super(EarlyStoppingWithTime, self).on_epoch_end(epoch=epoch, logs=logs)
        # check the running time
        if time.time() - self.start_time > self.seconds:
            self.model.stop_training = True
            if self.verbose > 0:
                print('Stopping after %s seconds.' % self.seconds)
            if self.restore_best_weights:
                self.model.set_weights(self.best_weights)


class CSVPDFLogger(CSVLogger):
    """Callback that streams epoch results to a csv file and a pdf plot

    """

    def __init__(self, pdf_filename=None, **kwargs):
        self.pdf_filename = pdf_filename
        super(CSVPDFLogger, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        super(CSVPDFLogger, self).on_epoch_end(epoch=epoch, logs=logs)
        self.plot()

    def plot(self):
        df = pd.read_csv(self.filename)
        plt.style.use("ggplot")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.plot(df['binary_accuracy'])
        ax1.plot(df['val_binary_accuracy'])
        ax1.set_title('Accuracy')
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'validation'], loc='lower right')

        ax2.plot(df['loss'])
        ax2.plot(df['val_loss'])
        ax2.set_title('Classification Loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['train', 'validation'], loc='upper right')

        fig.tight_layout()
        fig.savefig(self.pdf_filename)


class MultiGPUModelCheckpoint(Callback):
    """Additional keyword args are passed to ModelCheckpoint; see those docs for information on what args are accepted.
    https://github.com/TextpertAi/alt-model-checkpoint/blob/master/alt_model_checkpoint/__init__.py
    """

    def __init__(self,
                 filepath=None,
                 model_to_save=None,
                 best=np.Inf,
                 monitor='val_loss'):
        super(MultiGPUModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.model_to_save = model_to_save
        self.best = best
        self.monitor = monitor
        self.monitor_op = np.less

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        current = logs.get(self.monitor)

        if current is None:
            warnings.warn('Can save best model only with %s available, '
                          'skipping.' % self.monitor, RuntimeWarning)
        else:
            if self.monitor_op(current, self.best):
                print('\nEpoch %05d: %s improved from %0.8f to %0.8f,'
                      ' saving model to %s'
                      % (epoch + 1, self.monitor, self.best,
                         current, filepath), file=sys.stdout)
                self.best = current

                try:
                    f = h5dict(filepath, mode='w')
                    _serialize_model(self.model_to_save, f, include_optimizer=True)
                    f.close()
                except:
                    print("There is something wrong with saving model, will skip it", file=sys.stderr)

            else:
                print('\nEpoch %05d: %s did not improve from %0.8f' %
                      (epoch + 1, self.monitor, self.best), file=sys.stdout)


def build_callbacks(model=None,
                    weights_path=None,
                    history_path=None,
                    acc_loss_path=None,
                    exp_config=None):
    check_point_path = os.path.join(weights_path, "{}.h5".format(exp_config))
    history_filename = os.path.join(history_path, "{}.log".format(exp_config))
    pdf_filename = os.path.join(acc_loss_path, "{}.pdf".format(exp_config))

    best = np.inf
    # check if the log file exists and empty
    if os.path.exists(history_filename) and os.stat(history_filename).st_size > 0:
        # recover the best validation loss from logs
        df = pd.read_csv(history_filename)
        best = np.min(df['val_loss'])

    check_pointer = MultiGPUModelCheckpoint(model_to_save=model,
                                            best=best,
                                            filepath=check_point_path,
                                            monitor='val_loss')

    early_stopper = EarlyStoppingWithTime(seconds=3600 * 20,
                                          monitor='val_loss',
                                          patience=5,
                                          verbose=1,
                                          restore_best_weights=True)

    csv_pdf_logger = CSVPDFLogger(pdf_filename=pdf_filename,
                                  filename=history_filename,
                                  append=True)

    learning_rate = ReduceLROnPlateau(patience=2, min_lr=1e-05)

    callbacks = [check_pointer, early_stopper, csv_pdf_logger, learning_rate]

    return callbacks
