from __future__ import print_function
from __future__ import division

from sklearn import linear_model

from utils import *
from configure import *

NET_NAMES = ['ResNet50', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'InceptionV3', 'InceptionResNetV2']


def main():
    for i, net_name in enumerate(NET_NAMES):
        training_predicted_path = get_training_predict_path(net_name)
        filename = os.path.join(training_predicted_path, "{}.npz".format(net_name))
        assert os.path.exists(filename), "the prediction {} does not exist".format(filename)

        x_train = np.load(filename)['pred']
        y_train = np.load(filename)['label']

        test_predicted_path = get_test_predict_path(net_name)
        filename = os.path.join(test_predicted_path, "{}.npz".format(net_name))
        assert os.path.exists(filename), "the prediction {} does not exist".format(filename)

        x_test = np.load(filename)['pred']
        y_pred = np.zeros(shape=x_train.shape, dtype=np.float32)
        test_pred = np.zeros(shape=(N_TEST, N_LABELS), dtype=np.float32)

        for label in range(N_LABELS):
            reg = linear_model.Ridge(alpha=0.5, max_iter=1000)
            reg.fit(X=x_train, y=y_train[:, label])
            y_pred[:, label] = np.clip(reg.predict(x_train), 1e-08, 1 - 1e-08)
            test_pred[:, label] = np.clip(reg.predict(x_test), 1e-08, 1 - 1e-08)

        filename = os.path.join("/home/rs619065/HPAIC/training", net_name, "{}_reg.npz".format(net_name))
        np.savez(filename, label=y_train, cnn_pred=x_train, reg_pred=y_pred)

        filename = os.path.join("/home/rs619065/HPAIC/test", net_name, "{}_reg.npz".format(net_name))
        np.savez(filename, cnn_pred=x_test, reg_pred=test_pred)


if __name__ == '__main__':
    main()
