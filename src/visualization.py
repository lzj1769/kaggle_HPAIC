from __future__ import print_function, division
import os
import time
import datetime
import psutil
import threading
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configure import *
from utils import gpu_info


def visua_acc_loss(history, exp_id):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.plot(history.history['acc'])
    ax1.plot(history.history['val_acc'])
    ax1.set_title('Accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left')

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')

    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_ACC_LOSS_PATH, "{}_acc_loss.pdf".format(exp_id)))


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


class SysMonitor(threading.Thread):
    shutdown = False

    def __init__(self):
        threading.Thread.__init__(self)
        self.utils = []

    def run(self):
        while not self.shutdown:
            dt = datetime.datetime.now()
            util = gpu_info()
            cpu_percent = psutil.cpu_percent()
            self.utils.append([dt] + [x[2] for x in util] + [cpu_percent])
            time.sleep(.1)

    def stop(self):
        self.shutdown = True

    def plot(self, exp_id):
        fig, ax = plt.subplots(2, 1, figsize=(15, 6))

        ax[0].title.set_text('GPU Utilization')
        ax[0].plot([u[1] for u in self.utils])
        ax[0].set_ylim([0, 100])
        ax[1].title.set_text('CPU Utilization')
        ax[1].plot([u[2] for u in self.utils])
        ax[1].set_ylim([0, 100])

        fig.savefig(os.path.join(GPU_MONITOR_PATH, "{}.pdf".format(exp_id)))
