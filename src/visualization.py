from __future__ import print_function, division
import os
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize': (12, 10)})

from configure import *

import keras.backend as K
from keras.layers import Input, Conv2DTranspose
from keras import Model
from keras.initializers import Ones, Zeros
from PIL import Image


def visua_acc_loss(acc_loss_path, logs_path, exp_config):
    # load logs
    loss_filename = os.path.join(logs_path, "{}.log".format(exp_config))
    df = pd.read_csv(loss_filename)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.plot(df['binary_accuracy'])
    ax1.plot(df['val_binary_accuracy'])
    ax1.set_title('Accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left')

    ax2.plot(df['loss'])
    ax2.plot(df['val_loss'])
    ax2.set_title('Loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')

    fig.tight_layout()
    fig.savefig(os.path.join(acc_loss_path, "{}.pdf".format(exp_config)))


def visua_threshold_f1(f1_score, optimal_thresholds, exp_id):
    thresholds = np.linspace(0, 1, 100)
    optimal_thresholds = np.round(optimal_thresholds, 2)

    plt.style.use("default")
    fig, ax = plt.subplots(7, 4, figsize=(10, 12))
    for i in range(7):
        for j in range(4):
            if i * 4 + j < 28:
                ax[i][j].plot(thresholds, f1_score[i * 4 + j])
                ax[i][j].set_title("Class: {}, Threshold: {}".format(i * 4 + j, optimal_thresholds[i * 4 + j]))
                ax[i][j].set_ylim([0, 1])
            else:
                break

    fig.tight_layout()
    fig.savefig(os.path.join(VISUALIZATION_PATH, "{}_threshold_f1.pdf".format(exp_id)))


def visua_f1_classes(f1_score, exp_id):
    fig, ax = plt.subplots(figsize=(8, 8))
    ind = np.arange(N_LABELS)

    plt.bar(ind, f1_score)
    ax.set_xticks(ind)
    xlabels = []
    for i in np.arange(N_LABELS):
        xlabels.append("Class {}".format(i))

    ax.set_xticklabels(ind)
    ax.set_ylim([0, 1])
    ax.set_ylabel('F1 score')
    ax.set_title('Macro F1 score: {}'.format(np.mean(f1_score)))

    fig.tight_layout()
    fig.savefig(os.path.join(VISUALIZATION_PATH, "{}_f1_class.pdf".format(exp_id)))


def visua_prob_distribution(visua_path, exp_config, training_prob, test_prob):
    fig, ax = plt.subplots(7, 4, figsize=(10, 12))
    for i in range(7):
        for j in range(4):
            if i * 4 + j < 28:
                sns.kdeplot(data=training_prob[:, i * 4 + j],
                            label="Validation",
                            ax=ax[i][j],
                            shade=True,
                            color="r")
                sns.kdeplot(data=test_prob[:, i * 4 + j],
                            label="Test",
                            ax=ax[i][j],
                            shade=True,
                            color="b")
                ax[i][j].set_title("Class: {}".format(i * 4 + j))
            else:
                break

    fig.tight_layout()
    filename = os.path.join(visua_path, "{}.pdf".format(exp_config))
    fig.savefig(filename)


class VisualBackprop:
    """A SaliencyMask class that computes saliency masks with VisualBackprop (https://arxiv.org/abs/1611.05418).
    """

    def __init__(self, model):
        """Constructs a VisualProp SaliencyMask."""
        inp = model.input

        outputs = [layer.output for layer in model.layers]  # all layer outputs
        outputs = [output for output in outputs if 'input_' not in output.name]

        self.forward_pass = K.function([inp, K.learning_phase()], outputs)  # evaluation function
        self.model = model

    def get_mask(self, input_image):
        """Returns a VisualBackprop mask."""
        x_value = np.expand_dims(input_image, axis=0)
        visual_bpr = None
        layer_outs = self.forward_pass([x_value, 0.])

        for i in range(len(self.model.layers) - 1, -1, -1):
            if 'Conv2D' in str(type(self.model.layers[i])):
                layer = np.mean(layer_outs[i], axis=3, keepdims=True)
                layer = layer - np.min(layer)
                layer = layer / (np.max(layer) - np.min(layer) + 1e-6)

                if visual_bpr is not None:
                    if visual_bpr.shape != layer.shape:
                        visual_bpr = self._deconv(visual_bpr)
                    visual_bpr = visual_bpr * layer
                else:
                    visual_bpr = layer

        return visual_bpr[0]

    def get_smoothed_mask(self, input_image, stdev_spread=.2, nsamples=50):
        """Returns a mask that is smoothed with the SmoothGrad method.
        Args:
            input_image: input image with shape (H, W, 3).
        """
        stdev = stdev_spread * (np.max(input_image) - np.min(input_image))

        total_gradients = np.zeros_like(input_image)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, input_image.shape)
            x_value_plus_noise = input_image + noise

            total_gradients += self.get_mask(x_value_plus_noise)

        return total_gradients / nsamples

    def _deconv(self, feature_map):
        """The deconvolution operation to upsample the average feature map downstream"""
        x = Input(shape=(None, None, 1))
        y = Conv2DTranspose(filters=1,
                            kernel_size=(3, 3),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=Ones(),
                            bias_initializer=Zeros())(x)

        deconv_model = Model(inputs=[x], outputs=[y])

        inps = [deconv_model.input, K.learning_phase()]  # input placeholder
        outs = [deconv_model.layers[-1].output]  # output placeholder
        deconv_func = K.function(inps, outs)  # evaluation function

        return deconv_func([feature_map, 0])[0]


def save_image(image, mask, ax=None, title=''):
    if ax is None:
        plt.figure()
    plt.axis('off')

    vmax = np.percentile(image, 99)
    vmin = np.min(mask)

    plt.imsave("mask.png", mask, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
    plt.title(title)

    image = image.astype('uint8')
    plt.imsave("image.png", image)


def visua_cnn(model, image):
    visua_backprop = VisualBackprop(model=model)
    processed_image = image.astype(K.floatx())
    processed_image /= 128.
    processed_image -= 1.0

    mask = visua_backprop.get_mask(processed_image)
    mask += 1.0
    mask *= 128.

    save_image(image, mask)
