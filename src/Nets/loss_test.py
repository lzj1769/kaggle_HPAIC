from __future__ import print_function

import numpy as np
from loss import focal_loss
from loss import precision_recall_auc_loss

import tensorflow as tf


Y_true = np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
Y_pred = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1]])


loss = focal_loss(Y_true, Y_pred, alpha=0.5)

with tf.Session() as sess:
    print(sess.run(loss))