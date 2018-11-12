from __future__ import print_function

import os
import pandas as pd
import time

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import CSVLogger, ReduceLROnPlateau

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
        ax1.legend(['train', 'validation'], loc='upper left')

        ax2.plot(df['loss'])
        ax2.plot(df['val_loss'])
        ax2.set_title('Classification Loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['train', 'validation'], loc='upper left')

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

    def __init__(self, alternate_model, **kwargs):
        self.alternate_model = alternate_model
        super(AltModelCheckpoint, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        model_before = self.model
        self.model = self.alternate_model
        super(AltModelCheckpoint, self).on_epoch_end(epoch, logs)
        self.model = model_before


def build_callbacks(model, weights_path, logs_path, acc_loss_path, exp_config):
    fp = os.path.join(weights_path, "{}.h5".format(exp_config))
    check_pointer = AltModelCheckpoint(alternate_model=model,
                                       filepath=fp,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True)

    early_stopper = EarlyStoppingWithTime(seconds=3600 * 7,
                                          monitor='val_loss',
                                          patience=20,
                                          verbose=1,
                                          restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=4,
                                  min_lr=1e-06,
                                  min_delta=0.,
                                  verbose=1)

    filename = os.path.join(logs_path, "{}.log".format(exp_config))
    pdf_filename = os.path.join(acc_loss_path, "{}.pdf".format(exp_config))

    csv_pdf_logger = CSVPDFLogger(pdf_filename=pdf_filename,
                                  filename=filename,
                                  append=True)

    callbacks = [check_pointer, early_stopper, reduce_lr, csv_pdf_logger]

    return callbacks
