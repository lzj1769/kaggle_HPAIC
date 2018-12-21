from __future__ import print_function
import os

from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import StratifiedKFold
import numpy as np

from utils import get_ids, get_target
from configure import *


def multi_labels_split():
    ids = get_ids()
    target = get_target()

    stratifier = IterativeStratification(n_splits=K_FOLD, random_state=1769)

    for i, (train_indexes, test_indexes) in enumerate(stratifier.split(X=ids, y=target)):

        split_filename = os.path.join(DATA_DIR, "KFold_{}".format(i))
        np.savez(file=split_filename, train_indexes=train_indexes, test_indexes=test_indexes)


def single_label_split():
    ids = get_ids()
    target = get_target()

    for i in range(N_LABELS):
        labels = target[:, i]
        stratifier = StratifiedKFold(n_splits=SINGLE_LABEL_K_FOLD, random_state=1769)

        for j, (train_indexes, test_indexes) in enumerate(stratifier.split(X=ids, y=labels)):

            split_filename = os.path.join(DATA_DIR, "Label_{}_KFold_{}".format(i, j))
            np.savez(file=split_filename, train_indexes=train_indexes, test_indexes=test_indexes)


if __name__ == '__main__':
    single_label_split()
