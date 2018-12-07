from __future__ import print_function

import os
import numpy as np
import pandas as pd
import time

from keras.callbacks import ProgbarLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
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

    def plot_attention(self):
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
        fig.savefig(self.pdf_filename)


class AltModelCheckpoint(ModelCheckpoint):
    """Additional keyword args are passed to ModelCheckpoint; see those docs for information on what args are accepted.
    https://github.com/TextpertAi/alt-model-checkpoint/blob/master/alt_model_checkpoint/__init__.py
    """

    def __init__(self, alternate_model, best=np.Inf, **kwargs):
        self.alternate_model = alternate_model
        super(AltModelCheckpoint, self).__init__(**kwargs)
        self.best = best

    def on_epoch_end(self, epoch, logs=None):
        model_before = self.model
        self.model = self.alternate_model
        super(AltModelCheckpoint, self).on_epoch_end(epoch, logs)
        self.model = model_before


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

    check_pointer = AltModelCheckpoint(alternate_model=model,
                                       best=best,
                                       filepath=check_point_path,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True)

    early_stopper = EarlyStoppingWithTime(seconds=3600 * 46,
                                          monitor='val_loss',
                                          patience=20,
                                          verbose=1,
                                          restore_best_weights=True)

    csv_pdf_logger = CSVPDFLogger(pdf_filename=pdf_filename,
                                  filename=history_filename,
                                  append=True)

    batch_logger = BatchProgbarLogger(display=500,
                                      count_mode='steps',
                                      stateful_metrics=model.stateful_metric_names)

    callbacks = [check_pointer, early_stopper, csv_pdf_logger, batch_logger]

    return callbacks
