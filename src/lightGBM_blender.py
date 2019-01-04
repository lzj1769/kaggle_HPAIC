from __future__ import print_function
from __future__ import division

import sys
import argparse
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from utils import *
from configure import *

NET_NAMES = ['ResNet50', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'InceptionV3', 'InceptionResNetV2']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', "--verbose", type=int, default=2, help="Verbosity mode. DEFAULT: 2")
    return parser.parse_args()


def get_data():
    x_train = None
    x_test = None

    for net_name in NET_NAMES:
        training_predicted_path = get_training_predict_path(net_name)
        filename = os.path.join(training_predicted_path, "{}.npz".format(net_name))
        assert os.path.exists(filename), "the prediction {} does not exist".format(filename)

        if x_train is None:
            x_train = np.load(filename)['pred']

        else:
            x_train = np.append(x_train, np.load(filename)['pred'], axis=1)

        test_predicted_path = get_test_predict_path(net_name)
        filename = os.path.join(test_predicted_path, "{}.npz".format(net_name))
        assert os.path.exists(filename), "the prediction {} does not exist".format(filename)

        if x_test is None:
            x_test = np.load(filename)['pred']

        else:
            x_test = np.append(x_test, np.load(filename)['pred'], axis=1)

    y_train = get_target()

    return x_train, y_train, x_test


def main():
    x_train, y_train, x_test = get_data()

    params = dict()

    params["objective"] = "binary"
    params["boosting"] = "gbdt"
    params["metric"] = "binary_logloss"
    params["num_leaves"] = 31
    params["bagging_fraction"] = 0.8
    params["feature_fraction"] = 0.8
    params["learning_rate"] = 0.01
    params["max_depth"] = -1
    params["num_threads"] = 4
    params["bagging_freq"] = 20
    params["max_bin"] = 512

    x_pred = np.zeros(shape=(N_TRAINING, N_LABELS), dtype=np.float32)
    test_pred = np.zeros(shape=(N_TEST, N_LABELS), dtype=np.float32)

    for label in range(N_LABELS):
        skf = StratifiedKFold(n_splits=5, random_state=label)

        for i, (train_indexes, valid_indexes) in enumerate(skf.split(X=x_train, y=y_train[:, label])):

            print('Length of train: {}'.format(len(train_indexes)), file=sys.stdout)
            print('Length of validation: {}'.format(len(valid_indexes)), file=sys.stdout)

            # create dataset for lightgbm
            lgb_train = lgb.Dataset(x_train[train_indexes], y_train[train_indexes, label])
            lgb_eval = lgb.Dataset(x_train[valid_indexes], y_train[valid_indexes, label], reference=lgb_train)

            print('Starting training...')
            # train
            gbm = lgb.train(params,
                            lgb_train,
                            num_boost_round=2000,
                            valid_sets=lgb_eval,
                            early_stopping_rounds=50,
                            verbose_eval=1)

            print('Saving model...')
            # save model to file
            filename = os.path.join("/home/rs619065/HPAIC/model/LightGBM", "Label_{}_Kfold_{}.txt".format(label, i))
            gbm.save_model(filename)

            print('Starting predicting...')
            # predict
            y_pred = gbm.predict(x_train[valid_indexes], num_iteration=gbm.best_iteration)
            # eval
            print('The log loss of prediction is:', log_loss(y_train[valid_indexes, label], y_pred))

            for j, index in enumerate(valid_indexes):
                x_pred[index, label] = y_pred[j]

            test_pred[:, label] += gbm.predict(x_test, num_iteration=gbm.best_iteration)

    filename = "/home/rs619065/HPAIC/training/LightGBM/LightGBM.npz"
    np.savez(file=filename, pred=x_pred, label=y_train)

    test_pred /= 5
    filename = "/home/rs619065/HPAIC/test/LightGBM/LightGBM.npz"
    np.savez(file=filename, pred=test_pred)


if __name__ == '__main__':
    main()
