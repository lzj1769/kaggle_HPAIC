from __future__ import print_function
from __future__ import division

import os
import sys

import pandas as pd
import numpy as np

from generator import ImageDataGenerator
from configure import *
from utils import load_data


def predict(model, net_name, train_f1, val_f1, output_shape, n_channels,
            optimal_thresholds, preprocessing_function):

    print("predicting...", file=sys.stderr)

    test_img = load_data(dataset="test")
    test_generator = ImageDataGenerator(x=test_img, output_shape=output_shape,
                                        shuffle=False, augment=False,
                                        n_channels=n_channels,
                                        preprocessing_function=preprocessing_function)
    y_pred = model.predict_generator(test_generator)

    predict_filename = os.path.join(PREDICTION_PATH, "{}.npz".format(net_name))
    np.savez(predict_filename, pred=y_pred)

    df = pd.read_csv(SAMPLE_SUBMISSION)
    predicted = list()
    for i in range(df.shape[0]):
        label_predict = np.arange(N_LABELS)[np.greater(y_pred[i], optimal_thresholds)]
        if label_predict.size == 0:
            str_predict_label = " ".join(str(label) for label in np.arange(N_LABELS))
        else:
            str_predict_label = " ".join(str(label) for label in label_predict)
        predicted.append(str_predict_label)

    df['Predicted'] = predicted
    submission_filename = "{}_train_f1_{}_val_f1_{}.csv".format(net_name, train_f1, val_f1)
    filename = os.path.join(SUBMISSION_PATH, submission_filename)
    df.to_csv(filename, index=False)
