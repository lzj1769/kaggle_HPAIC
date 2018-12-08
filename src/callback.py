from __future__ import print_function

import os
import numpy as np
import pandas as pd
import time
import warnings

from keras.callbacks import ProgbarLogger, Callback
from keras.callbacks import EarlyStopping, CSVLogger
from keras.utils.generic_utils import Progbar
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
                 save_best_only=True,
                 monitor='val_loss',
                 verbose=0):
        super(MultiGPUModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.model_to_save = model_to_save
        self.best = best
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.monitor_op = np.less

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch + 1, self.monitor, self.best,
                                 current, filepath))
                    self.best = current
                    self.model_to_save.save(filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve from %0.5f' %
                              (epoch + 1, self.monitor, self.best))


class BatchProgbarLogger(ProgbarLogger):

    def __init__(self, display=1, **kwargs):
        self.display = display
        super(BatchProgbarLogger, self).__init__(**kwargs)
        self.target = None
        self.progbar = None
        self.seen = None

    def on_epoch_begin(self, epoch, logs=None):
        print('Epoch %d/%d' % (epoch + 1, self.epochs))
        if self.use_steps:
            target = self.params['steps']
        else:
            target = self.params['samples']
        self.target = target
        self.progbar = Progbar(target=self.target,
                               verbose=1,
                               stateful_metrics=self.stateful_metrics)

        self.seen = 0

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        if self.use_steps:
            self.seen += 1
        else:
            self.seen += batch_size

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.seen < self.target and self.seen % self.display == 0:
            self.progbar.update(self.seen, self.log_values)


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
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True)

    early_stopper = EarlyStoppingWithTime(seconds=3600 * 110,
                                          monitor='val_loss',
                                          patience=20,
                                          verbose=1,
                                          restore_best_weights=True)

    csv_pdf_logger = CSVPDFLogger(pdf_filename=pdf_filename,
                                  filename=history_filename,
                                  append=True)

    # batch_logger = BatchProgbarLogger(display=500,
    #                                   count_mode='steps',
    #                                   stateful_metrics=model.stateful_metric_names)

    callbacks = [check_pointer, early_stopper, csv_pdf_logger]

    return callbacks
