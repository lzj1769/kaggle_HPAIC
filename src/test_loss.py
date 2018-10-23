import tensorflow as tf
import keras.backend as K

import numpy as np
import math

model_out = tf.constant([[1.0, 0.0, 0.0, 0.00, 0.0]], dtype=tf.float32)

labels = tf.constant([[0, 0, 0, 0, 0]], dtype=tf.float32)


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)


def binary_crossentropy(target, output, from_logits=True):
    """Binary crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.

    # Returns
        A tensor.
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        _epsilon = _to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))

    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                   logits=output)


def my_binary_crossentropy(y_true, y_pred):
    result = []
    for i in range(len(y_pred)):
        y_pred[i] = [max(min(x, 1 - 1e-08), 1e-08) for x in y_pred[i]]
        result.append(-np.mean([y_true[i][j] * math.log(y_pred[i][j]) +
                                (1 - y_true[i][j]) * math.log(1 - y_pred[i][j]) for j in range(len(y_pred[i]))]))
    return result


loss = binary_crossentropy(labels, model_out)


model_out = np.array([[1.0, 0.0, 0.0, 0.00, 0.0],
                      [0.0, 0.0, 0.0, 0.95, 0.75]])

labels = np.array([[0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 1]])

my_loss = my_binary_crossentropy(labels, model_out)


with tf.Session() as sess:
    print(sess.run(loss))
    print(my_loss)