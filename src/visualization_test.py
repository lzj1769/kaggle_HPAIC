from __future__ import print_function, division
import numpy as np

from configure import *

from visualization import visua_threshold_f1
from visualization import visua_f1_classes


def test_visua_threshold_f1():
    f1_score = list()
    optimal_threshold = list()
    for i in range(N_LABELS):
        f1_score.append(np.random.rand(100))
        optimal_threshold.append(np.random.rand())

    visua_threshold_f1(f1_score,  optimal_threshold, "test")


def test_visua_f1_classes():
    f1_score = np.random.rand(N_LABELS)
    visua_f1_classes(f1_score, "test")


def test_visua_prob_distribution():
    training_prob