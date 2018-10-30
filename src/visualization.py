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


def visua_acc_loss(acc_loss_path, logs_path, exp_config):
    # load logs
    loss_filename = os.path.join(logs_path, "{}.log".format(exp_config))
    df = pd.read_csv(loss_filename)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.plot(df['acc'])
    ax1.plot(df['val_acc'])
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
    training_prob_reshape = training_prob.flatten().tolist()
    test_prob_reshape = test_prob.flatten().tolist()

    classes = np.arange(N_LABELS).tolist()
    training_classes = classes * training_prob.shape[0]
    test_classes = classes * test_prob.shape[0]

    training_labels = ['Training'] * len(training_prob_reshape)
    test_labels = ['Test'] * len(test_prob_reshape)

    plot_data = dict()
    plot_data['Prob'] = training_prob_reshape + test_prob_reshape
    plot_data['Classes'] = training_classes + test_classes
    plot_data['Dataset'] = training_labels + test_labels

    df = pd.DataFrame(data=plot_data)

    box_plot = sns.boxplot(x="Classes", y="Prob", hue="Dataset", data=df, fliersize=0.1, notch=True)

    box_plot.legend()
    fig = box_plot.get_figure()
    fig.tight_layout()
    filename = os.path.join(visua_path, "{}.png".format(exp_config))
    fig.savefig(filename)
