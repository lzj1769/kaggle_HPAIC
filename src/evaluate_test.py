from __future__ import print_function
import os

from utils import *

from keras.models import load_model
import Nets.PreTrained_MobileNet_LogLoss as net
from evaluate import evaluate

epochs = net.epochs
batch_size = net.batch_size
callbacks = net.build_callbacks(MODEL_PATH, "PreTrained_MobileNet_LogLoss")
input_shape = net.INPUT_SHAPE

weight_filename = os.path.join(MODEL_PATH, "{}.h5".format("PreTrained_MobileNet_LogLoss"))
model = load_model(weight_filename)

if hasattr(net, 'preprocess_input'):
    preprocessing_function = net.preprocess_input
else:
    preprocessing_function = None


x_train, y_train = load_data(dataset="train")
x_valid, y_valid = load_data(dataset="validation")

train_f1, val_f1, optimal_thresholds = evaluate(model, "PreTrained_MobileNet_LogLoss",
                                                x_train, y_train, x_valid, y_valid,
                                                output_shape=(input_shape[0], input_shape[1]),
                                                n_channels=input_shape[2],
                                                preprocessing_function=preprocessing_function)

