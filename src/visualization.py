from __future__ import print_function, division
import os
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

sns.set(rc={'figure.figsize': (12, 10)})

from configure import *


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


def visua_prob_distribution(visua_path, net_name, training_prob, test_prob):
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
    filename = os.path.join(visua_path, "{}.pdf".format(net_name))
    fig.savefig(filename)


def visua_cnn(model, image=None, id=id):
    from vis.utils.utils import find_layer_idx
    from keras import activations
    from vis.visualization import visualize_cam
    from vis.utils.utils import apply_modifications

    # Utility to search for layer index by name.
    # Alternatively we can specify this as -1 since it corresponds to the last layer.

    penultimate_layer = find_layer_idx(model, 'res5b_branch2b')

    # Swap softmax with linear
    model.layers[penultimate_layer].activation = activations.linear
    model = apply_modifications(model)

    grads = visualize_cam(model, penultimate_layer, filter_indices=None,
                          seed_input=image, backprop_modifier='guided')

    fig = plt.figure(dpi=300, tight_layout=True)
    fig.figimage(np.uint8(grads[:, :, :3]), vmin=0, vmax=255, cmap='jet')
    fig.figimage(image[:, :, :3], xo=1024)

    DPI = fig.get_dpi()
    fig.set_size_inches(2 * 1024.0 / float(DPI), 1024.0 / float(DPI))
    fig.savefig("{}.png".format(id))
