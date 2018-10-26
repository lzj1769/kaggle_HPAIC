from __future__ import print_function
import os

from skmultilearn.model_selection import IterativeStratification
import numpy as np

from utils import load_data
from configure import *


def main():
    img, labels = load_data(dataset="train")

    stratifier = IterativeStratification(n_splits=8, random_state=1769)

    for i, (train_indexes, test_indexes) in enumerate(stratifier.split(X=img, y=labels)):
        print(train_indexes)
        print(test_indexes)

        split_filename = os.path.join(DATA_DIR, "KFold_{}".format(i))
        np.savez(file=split_filename, train_indexes=train_indexes, test_indexes=test_indexes)


if __name__ == '__main__':
    main()
