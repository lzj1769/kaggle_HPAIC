from __future__ import print_function
from __future__ import division

import sys
import time
import argparse
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

from utils import *
from configure import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', "--label", type=int, default=None, help="which label to predict")
    parser.add_argument('-r', "--runs", type=int, default=100, help="number of runs for each label. DEFAULT: 100")
    parser.add_argument('-v', "--verbose", type=int, default=2, help="Verbosity mode. DEFAULT: 2")
    return parser.parse_args()


def load_ensemble_data():
    print('Loading data...', file=sys.stdout)

    ensemble_training_path = os.path.join(TRAINING_OUTPUT_PATH, "Ensemble")
    filename = os.path.join(ensemble_training_path, "Ensemble.npz")
    training_data = np.load(filename)

    ensemble_test_path = os.path.join(TEST_OUTPUT_PATH, "Ensemble")
    filename = os.path.join(ensemble_test_path, "Ensemble.npz")
    test_data = np.load(filename)

    return training_data, test_data


def gbm_blender(train_indexes, test_indexes, params, label):
    training_data, test_data = load_ensemble_data()
    x = training_data['pred']
    y = training_data['label'][:, label]

    print(label)

    num_folds = len(train_indexes)
    x_pred = np.zeros(shape=(x.shape[0], 1), dtype=np.float32)
    for i in range(num_folds):
        train_index = train_indexes[i]
        test_index = test_indexes[i]

        x_train = x[train_index]
        y_train = y[train_index]
        x_valid = x[test_index]
        y_valid = y[test_index]

        print('Length of train: ', x_train.shape[0])
        print('Length of validation: ', x_valid.shape[0])

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)

        print('Starting training...')
        # train
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=2000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=100)

        print('Saving model...')
        # save model to file
        gbm.save_model('model.txt')

        print('Starting predicting...')
        # predict
        y_pred = gbm.predict(x_valid, num_iteration=gbm.best_iteration)
        # eval
        print('The log loss of prediction is:', log_loss(y_valid, y_pred))

        for i, index in enumerate(test_index):
            x_pred[index] = y_pred[i]

    np.savez(file="test.npz", pred=x_pred, label=y)

    max_f1 = -np.inf
    optimal_threshold = None
    thresholds = np.linspace(0, 1, 1000)
    for threshold in thresholds:
        f1 = f1_score(y, x_pred > threshold)
        if f1 > max_f1:
            max_f1 = f1
            optimal_threshold = threshold

    print(max_f1)
    print(optimal_threshold)


def gbm_random_step(label, run):
    random_state = int(time.time()) + label + run
    num_folds = 5

    training_data, test_data = load_ensemble_data()
    X = training_data['pred']

    train_indexes = list()
    test_indexes = list()
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    for train_index, test_index in kf.split(list(range(X.shape[0]))):
        train_indexes.append(train_index)
        test_indexes.append(test_index)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 31,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1
    }

    return train_indexes, test_indexes, params


def main():
    args = parse_args()

    gbm_training_path = os.path.join(TRAINING_OUTPUT_PATH, "LightGBM")

    if not os.path.exists(gbm_training_path):
        os.mkdir(gbm_training_path)

    gbm_test_path = os.path.join(TEST_OUTPUT_PATH, "LightGBM")

    if not os.path.exists(gbm_test_path):
        os.mkdir(gbm_test_path)

    for r in range(args.runs):
        train_indexes, test_indexes, params = gbm_random_step(label=args.label, run=r)
        gbm_blender(train_indexes, test_indexes, params, label=args.label)


if __name__ == '__main__':
    main()
