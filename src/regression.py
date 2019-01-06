from __future__ import print_function
from __future__ import division

from sklearn import linear_model

from utils import *
from configure import *

NET_NAMES = ['ResNet50', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'InceptionV3', 'InceptionResNetV2']


def get_data():
    x_train = np.empty(shape=(N_TRAINING, N_LABELS, len(NET_NAMES)), dtype=np.float32)
    x_test = np.empty(shape=(N_TEST, N_LABELS, len(NET_NAMES)), dtype=np.float32)

    for i, net_name in enumerate(NET_NAMES):
        training_predicted_path = get_training_predict_path(net_name)
        filename = os.path.join(training_predicted_path, "{}.npz".format(net_name))
        assert os.path.exists(filename), "the prediction {} does not exist".format(filename)

        x_train[:, :, i] = np.load(filename)['pred']

        test_predicted_path = get_test_predict_path(net_name)
        filename = os.path.join(test_predicted_path, "{}.npz".format(net_name))
        assert os.path.exists(filename), "the prediction {} does not exist".format(filename)

        x_test[:, :, i] = np.load(filename)['pred']

    y_train = get_target()

    return x_train, y_train, x_test


def main():
    x_train, y_train, x_test = get_data()

    print(x_train.shape)
    print(y_train.shape)
    reg = linear_model.RidgeClassifier(alpha=.5, copy_X=True)

    x_pred = np.zeros(shape=(N_TRAINING, N_LABELS), dtype=np.float32)
    test_pred = np.zeros(shape=(N_TEST, N_LABELS), dtype=np.float32)

    reg.fit(X=x_train[:, 0, :], y=y_train[:, 0])


if __name__ == '__main__':
    main()
