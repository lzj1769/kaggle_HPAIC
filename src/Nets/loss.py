import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np
import keras.backend as K


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     y_pred: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     y_true: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    # transform back to logits
    zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(y_true > zeros, y_true - y_pred, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(y_true > zeros, zeros, y_pred)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(y_pred, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - y_pred, 1e-8, 1.0))

    return K.mean(tf.reduce_sum(per_entry_cross_ent, axis=-1), axis=-1)


def precision_recall_auc_loss(labels, logits, precision_range=(0.0, 1.0),
                              num_anchors=20, weights=1.0,
                              dual_rate_factor=0.1, label_priors=None,
                              surrogate_type='xent',
                              lambdas_initializer=tf.constant_initializer(1.0),
                              reuse=None, variables_collections=None,
                              trainable=True, scope=None):
    """Computes precision-recall AUC loss.

    The loss is based on a sum of losses for recall at a range of
    precision values (anchor points). This sum is a Riemann sum that
    approximates the area under the precision-recall curve.

    The per-example `weights` argument changes not only the coefficients of
    individual training examples, but how the examples are counted toward the
    constraint. If `label_priors` is given, it MUST take `weights` into account.
    That is,
        label_priors = P / (P + N)
    where
        P = sum_i (wt_i on positives)
        N = sum_i (wt_i on negatives).

    Args:
      labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
      logits: A `Tensor` with the same shape as `labels`.
      precision_range: A length-two tuple, the range of precision values over
        which to compute AUC. The entries must be nonnegative, increasing, and
        less than or equal to 1.0.
      num_anchors: The number of grid points used to approximate the Riemann sum.
      weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
        [batch_size] or [batch_size, num_labels].
      dual_rate_factor: A floating point value which controls the step size for
        the Lagrange multipliers.
      label_priors: None, or a floating point `Tensor` of shape [num_labels]
        containing the prior probability of each label (i.e. the fraction of the
        training data consisting of positive examples). If None, the label
        priors are computed from `labels` with a moving average. See the notes
        above regarding the interaction with `weights` and do not set this unless
        you have a good reason to do so.
      surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
        should be used for indicator functions.
      lambdas_initializer: An initializer for the Lagrange multipliers.
      reuse: Whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: Optional list of collections for the variables.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      scope: Optional scope for `variable_scope`.

    Returns:
      loss: A `Tensor` of the same shape as `logits` with the component-wise
        loss.
      other_outputs: A dictionary of useful internal quantities for debugging. For
        more details, see http://arxiv.org/pdf/1608.04802.pdf.
        lambdas: A Tensor of shape [1, num_labels, num_anchors] consisting of the
          Lagrange multipliers.
        biases: A Tensor of shape [1, num_labels, num_anchors] consisting of the
          learned bias term for each.
        label_priors: A Tensor of shape [1, num_labels, 1] consisting of the prior
          probability of each label learned by the loss, if not provided.
        true_positives_lower_bound: Lower bound on the number of true positives
          given `labels` and `logits`. This is the same lower bound which is used
          in the loss expression to be optimized.
        false_positives_upper_bound: Upper bound on the number of false positives
          given `labels` and `logits`. This is the same upper bound which is used
          in the loss expression to be optimized.

    Raises:
      ValueError: If `surrogate_type` is not `xent` or `hinge`.
    """
    with tf.variable_scope(scope,
                           'precision_recall_auc',
                           [labels, logits, label_priors],
                           reuse=reuse):
        labels, logits, weights, original_shape = _prepare_labels_logits_weights(
            labels, logits, weights)
        num_labels = get_num_labels(logits)

        # Convert other inputs to tensors and standardize dtypes.
        dual_rate_factor = convert_and_cast(
            dual_rate_factor, 'dual_rate_factor', logits.dtype)

        # Create Tensor of anchor points and distance between anchors.
        precision_values, delta = _range_to_anchors_and_delta(
            precision_range, num_anchors, logits.dtype)
        # Create lambdas with shape [1, num_labels, num_anchors].
        lambdas, lambdas_variable = _create_dual_variable(
            'lambdas',
            shape=[1, num_labels, num_anchors],
            dtype=logits.dtype,
            initializer=lambdas_initializer,
            collections=variables_collections,
            trainable=trainable,
            dual_rate_factor=dual_rate_factor)
        # Create biases with shape [1, num_labels, num_anchors].
        biases = tf.contrib.framework.model_variable(
            name='biases',
            shape=[1, num_labels, num_anchors],
            dtype=logits.dtype,
            initializer=tf.zeros_initializer(),
            collections=variables_collections,
            trainable=trainable)
        # Maybe create label_priors.
        label_priors = maybe_create_label_priors(
            label_priors, labels, weights, variables_collections)
        label_priors = tf.reshape(label_priors, [1, num_labels, 1])

        # Expand logits, labels, and weights to shape [batch_size, num_labels, 1].
        logits = tf.expand_dims(logits, 2)
        labels = tf.expand_dims(labels, 2)
        weights = tf.expand_dims(weights, 2)

        # Calculate weighted loss and other outputs. The log(2.0) term corrects for
        # logloss not being an upper bound on the indicator function.
        loss = weights * weighted_surrogate_loss(
            labels,
            logits + biases,
            surrogate_type=surrogate_type,
            positive_weights=1.0 + lambdas * (1.0 - precision_values),
            negative_weights=lambdas * precision_values)
        maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
        maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
        lambda_term = lambdas * (1.0 - precision_values) * label_priors * maybe_log2
        per_anchor_loss = loss - lambda_term
        per_label_loss = delta * tf.reduce_sum(per_anchor_loss, 2)
        # Normalize the AUC such that a perfect score function will have AUC 1.0.
        # Because precision_range is discretized into num_anchors + 1 intervals
        # but only num_anchors terms are included in the Riemann sum, the
        # effective length of the integration interval is `delta` less than the
        # length of precision_range.
        scaled_loss = tf.div(per_label_loss,
                             precision_range[1] - precision_range[0] - delta,
                             name='AUC_Normalize')
        scaled_loss = tf.reshape(scaled_loss, original_shape)

        other_outputs = {
            'lambdas': lambdas_variable,
            'biases': biases,
            'label_priors': label_priors,
            'true_positives_lower_bound': true_positives_lower_bound(
                labels, logits, weights, surrogate_type),
            'false_positives_upper_bound': false_positives_upper_bound(
                labels, logits, weights, surrogate_type)}

        return scaled_loss


def get_num_labels(labels_or_logits):
    """Returns the number of labels inferred from labels_or_logits."""
    if labels_or_logits.get_shape().ndims <= 1:
        return 1
    return labels_or_logits.get_shape()[1].value


def _create_dual_variable(name, shape, dtype, initializer, collections,
                          trainable, dual_rate_factor):
    """Creates a new dual variable.

    Dual variables are required to be nonnegative. If trainable, their gradient
    is reversed so that they are maximized (rather than minimized) by the
    optimizer.

    Args:
      name: A string, the name for the new variable.
      shape: Shape of the new variable.
      dtype: Data type for the new variable.
      initializer: Initializer for the new variable.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
      dual_rate_factor: A floating point value or `Tensor`. The learning rate for
        the dual variable is scaled by this factor.

    Returns:
      dual_value: An op that computes the absolute value of the dual variable
        and reverses its gradient.
      dual_variable: The underlying variable itself.
    """
    # We disable partitioning while constructing dual variables because they will
    # be updated with assign, which is not available for partitioned variables.
    partitioner = tf.get_variable_scope().partitioner
    try:
        tf.get_variable_scope().set_partitioner(None)
        dual_variable = tf.contrib.framework.model_variable(
            name=name,
            shape=shape,
            dtype=dtype,
            initializer=initializer,
            collections=collections,
            trainable=trainable)
    finally:
        tf.get_variable_scope().set_partitioner(partitioner)
    # Using the absolute value enforces nonnegativity.
    dual_value = tf.abs(dual_variable)

    if trainable:
        # To reverse the gradient on the dual variable, multiply the gradient by
        # -dual_rate_factor
        dual_value = (tf.stop_gradient((1.0 + dual_rate_factor) * dual_value)
                      - dual_rate_factor * dual_value)
    return dual_value, dual_variable


def true_positives_lower_bound(labels, logits, weights, surrogate_type):
    """Calculate a lower bound on the number of true positives.

    This lower bound on the number of true positives given `logits` and `labels`
    is the same one used in the global objectives loss functions.

    Args:
      labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
      logits: A `Tensor` of shape [batch_size, num_labels] or
        [batch_size, num_labels, num_anchors]. If the third dimension is present,
        the lower bound is computed on each slice [:, :, k] independently.
      weights: Per-example loss coefficients, with shape broadcast-compatible with
          that of `labels`.
      surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
        should be used for indicator functions.

    Returns:
      A `Tensor` of shape [num_labels] or [num_labels, num_anchors].
    """
    maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
    maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
    if logits.get_shape().ndims == 3 and labels.get_shape().ndims < 3:
        labels = tf.expand_dims(labels, 2)
    loss_on_positives = weighted_surrogate_loss(
        labels, logits, surrogate_type, negative_weights=0.0) / maybe_log2
    return tf.reduce_sum(weights * (labels - loss_on_positives), 0)


def false_positives_upper_bound(labels, logits, weights, surrogate_type):
    """Calculate an upper bound on the number of false positives.

    This upper bound on the number of false positives given `logits` and `labels`
    is the same one used in the global objectives loss functions.

    Args:
      labels: A `Tensor` of shape [batch_size, num_labels]
      logits: A `Tensor` of shape [batch_size, num_labels]  or
        [batch_size, num_labels, num_anchors]. If the third dimension is present,
        the lower bound is computed on each slice [:, :, k] independently.
      weights: Per-example loss coefficients, with shape broadcast-compatible with
          that of `labels`.
      surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
        should be used for indicator functions.

    Returns:
      A `Tensor` of shape [num_labels] or [num_labels, num_anchors].
    """
    maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
    maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
    loss_on_negatives = weighted_surrogate_loss(
        labels, logits, surrogate_type, positive_weights=0.0) / maybe_log2
    return tf.reduce_sum(weights * loss_on_negatives, 0)


def maybe_create_label_priors(label_priors,
                              labels,
                              weights,
                              variables_collections):
    """Creates moving average ops to track label priors, if necessary.

    Args:
      label_priors: As required in e.g. precision_recall_auc_loss.
      labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
      weights: As required in e.g. precision_recall_auc_loss.
      variables_collections: Optional list of collections for the variables, if
        any must be created.

    Returns:
      label_priors: A Tensor of shape [num_labels] consisting of the
        weighted label priors, after updating with moving average ops if created.
    """
    if label_priors is not None:
        label_priors = convert_and_cast(
            label_priors, name='label_priors', dtype=labels.dtype.base_dtype)
        return tf.squeeze(label_priors)

    label_priors = build_label_priors(
        labels,
        weights,
        variables_collections=variables_collections)
    return label_priors


def weighted_surrogate_loss(labels,
                            logits,
                            surrogate_type='xent',
                            positive_weights=1.0,
                            negative_weights=1.0,
                            name=None):
    """Returns either weighted cross-entropy or hinge loss.

    For example `surrogate_type` is 'xent' returns the weighted cross
    entropy loss.

    Args:
     labels: A `Tensor` of type `float32` or `float64`. Each entry must be
        between 0 and 1. `labels` can be a 2D tensor with shape
        [batch_size, num_labels] or a 3D tensor with shape
        [batch_size, num_labels, K].
      logits: A `Tensor` of the same type and shape as `labels`. If `logits` has
        shape [batch_size, num_labels, K], each slice [:, :, k] represents an
        'attempt' to predict `labels` and the loss is computed per slice.
      surrogate_type: A string that determines which loss to return, supports
      'xent' for cross-entropy and 'hinge' for hinge loss.
      positive_weights: A `Tensor` that holds positive weights and has the
        following semantics according to its shape:
          scalar - A global positive weight.
          1D tensor - must be of size K, a weight for each 'attempt'
          2D tensor - of size [num_labels, K'] where K' is either K or 1.
        The `positive_weights` will be expanded to the left to match the
        dimensions of logits and labels.
      negative_weights: A `Tensor` that holds positive weight and has the
        semantics identical to positive_weights.
      name: A name for the operation (optional).

    Returns:
      The weigthed loss.

    Raises:
      ValueError: If value of `surrogate_type` is not supported.
    """
    with tf.name_scope(
            name, 'weighted_loss',
            [logits, labels, surrogate_type, positive_weights,
             negative_weights]) as name:
        if surrogate_type == 'xent':
            return weighted_sigmoid_cross_entropy_with_logits(
                logits=logits,
                labels=labels,
                positive_weights=positive_weights,
                negative_weights=negative_weights,
                name=name)
        elif surrogate_type == 'hinge':
            return weighted_hinge_loss(
                logits=logits,
                labels=labels,
                positive_weights=positive_weights,
                negative_weights=negative_weights,
                name=name)
        raise ValueError('surrogate_type %s not supported.' % surrogate_type)


def weighted_sigmoid_cross_entropy_with_logits(labels,
                                               logits,
                                               positive_weights=1.0,
                                               negative_weights=1.0,
                                               name=None):
    """Computes a weighting of sigmoid cross entropy given `logits`.

    Measures the weighted probability error in discrete classification tasks in
    which classes are independent and not mutually exclusive.  For instance, one
    could perform multilabel classification where a picture can contain both an
    elephant and a dog at the same time. The class weight multiplies the
    different types of errors.
    For brevity, let `x = logits`, `z = labels`, `c = positive_weights`,
    `d = negative_weights`  The
    weighed logistic loss is

    ```
    c * z * -log(sigmoid(x)) + d * (1 - z) * -log(1 - sigmoid(x))
    = c * z * -log(1 / (1 + exp(-x))) - d * (1 - z) * log(exp(-x) / (1 + exp(-x)))
    = c * z * log(1 + exp(-x)) + d * (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
    = c * z * log(1 + exp(-x)) + d * (1 - z) * (x + log(1 + exp(-x)))
    = (1 - z) * x * d + (1 - z + c * z ) * log(1 + exp(-x))
    =  - d * x * z + d * x + (d - d * z + c * z ) * log(1 + exp(-x))
    ```

    To ensure stability and avoid overflow, the implementation uses the identity
        log(1 + exp(-x)) = max(0,-x) + log(1 + exp(-abs(x)))
    and the result is computed as

      ```
      = -d * x * z + d * x
        + (d - d * z + c * z ) * (max(0,-x) + log(1 + exp(-abs(x))))
      ```

    Note that the loss is NOT an upper bound on the 0-1 loss, unless it is divided
    by log(2).

    Args:
      labels: A `Tensor` of type `float32` or `float64`. `labels` can be a 2D
        tensor with shape [batch_size, num_labels] or a 3D tensor with shape
        [batch_size, num_labels, K].
      logits: A `Tensor` of the same type and shape as `labels`. If `logits` has
        shape [batch_size, num_labels, K], the loss is computed separately on each
        slice [:, :, k] of `logits`.
      positive_weights: A `Tensor` that holds positive weights and has the
        following semantics according to its shape:
          scalar - A global positive weight.
          1D tensor - must be of size K, a weight for each 'attempt'
          2D tensor - of size [num_labels, K'] where K' is either K or 1.
        The `positive_weights` will be expanded to the left to match the
        dimensions of logits and labels.
      negative_weights: A `Tensor` that holds positive weight and has the
        semantics identical to positive_weights.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of the same shape as `logits` with the componentwise
        weighted logistic losses.
    """
    with tf.name_scope(
            name,
            'weighted_logistic_loss',
            [logits, labels, positive_weights, negative_weights]) as name:
        labels, logits, positive_weights, negative_weights = prepare_loss_args(
            labels, logits, positive_weights, negative_weights)

        softplus_term = tf.add(tf.maximum(-logits, 0.0),
                               tf.log(1.0 + tf.exp(-tf.abs(logits))))
        weight_dependent_factor = (
                negative_weights + (positive_weights - negative_weights) * labels)
        return (negative_weights * (logits - labels * logits) +
                weight_dependent_factor * softplus_term)


def weighted_hinge_loss(labels,
                        logits,
                        positive_weights=1.0,
                        negative_weights=1.0,
                        name=None):
    """Computes weighted hinge loss given logits `logits`.

    The loss applies to multi-label classification tasks where labels are
    independent and not mutually exclusive. See also
    `weighted_sigmoid_cross_entropy_with_logits`.

    Args:
      labels: A `Tensor` of type `float32` or `float64`. Each entry must be
        either 0 or 1. `labels` can be a 2D tensor with shape
        [batch_size, num_labels] or a 3D tensor with shape
        [batch_size, num_labels, K].
      logits: A `Tensor` of the same type and shape as `labels`. If `logits` has
        shape [batch_size, num_labels, K], the loss is computed separately on each
        slice [:, :, k] of `logits`.
      positive_weights: A `Tensor` that holds positive weights and has the
        following semantics according to its shape:
          scalar - A global positive weight.
          1D tensor - must be of size K, a weight for each 'attempt'
          2D tensor - of size [num_labels, K'] where K' is either K or 1.
        The `positive_weights` will be expanded to the left to match the
        dimensions of logits and labels.
      negative_weights: A `Tensor` that holds positive weight and has the
        semantics identical to positive_weights.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of the same shape as `logits` with the componentwise
        weighted hinge loss.
    """
    with tf.name_scope(
            name, 'weighted_hinge_loss',
            [logits, labels, positive_weights, negative_weights]) as name:
        labels, logits, positive_weights, negative_weights = prepare_loss_args(
            labels, logits, positive_weights, negative_weights)

        positives_term = positive_weights * labels * tf.maximum(1.0 - logits, 0)
        negatives_term = (negative_weights * (1.0 - labels)
                          * tf.maximum(1.0 + logits, 0))
        return positives_term + negatives_term


def prepare_loss_args(labels, logits, positive_weights, negative_weights):
    """Prepare arguments for weighted loss functions.

    If needed, will convert given arguments to appropriate type and shape.

    Args:
      labels: labels or labels of the loss function.
      logits: Logits of the loss function.
      positive_weights: Weight on the positive examples.
      negative_weights: Weight on the negative examples.

    Returns:
      Converted labels, logits, positive_weights, negative_weights.
    """
    logits = tf.convert_to_tensor(logits, name='logits')
    labels = convert_and_cast(labels, 'labels', logits.dtype)
    if len(labels.get_shape()) == 2 and len(logits.get_shape()) == 3:
        labels = tf.expand_dims(labels, [2])

    positive_weights = convert_and_cast(positive_weights, 'positive_weights',
                                        logits.dtype)
    positive_weights = expand_outer(positive_weights, logits.get_shape().ndims)
    negative_weights = convert_and_cast(negative_weights, 'negative_weights',
                                        logits.dtype)
    negative_weights = expand_outer(negative_weights, logits.get_shape().ndims)
    return labels, logits, positive_weights, negative_weights


def convert_and_cast(value, name, dtype):
    """Convert input to tensor and cast to dtype.

    Args:
      value: An object whose type has a registered Tensor conversion function,
          e.g. python numerical type or numpy array.
      name: Name to use for the new Tensor, if one is created.
      dtype: Optional element type for the returned tensor.

    Returns:
      A tensor.
    """
    return tf.cast(tf.convert_to_tensor(value, name=name), dtype=dtype)


def expand_outer(tensor, rank):
    """Expands the given `Tensor` outwards to a target rank.

    For example if rank = 3 and tensor.shape is [3, 4], this function will expand
    to such that the resulting shape will be  [1, 3, 4].

    Args:
      tensor: The tensor to expand.
      rank: The target dimension.

    Returns:
      The expanded tensor.

    Raises:
      ValueError: If rank of `tensor` is unknown, or if `rank` is smaller than
        the rank of `tensor`.
    """
    if tensor.get_shape().ndims is None:
        raise ValueError('tensor dimension must be known.')
    if len(tensor.get_shape()) > rank:
        raise ValueError(
            '`rank` must be at least the current tensor dimension: (%s vs %s).' %
            (rank, len(tensor.get_shape())))
    while len(tensor.get_shape()) < rank:
        tensor = tf.expand_dims(tensor, 0)
    return tensor


def build_label_priors(labels,
                       weights=None,
                       positive_pseudocount=1.0,
                       negative_pseudocount=1.0,
                       variables_collections=None):
    """Creates an op to maintain and update label prior probabilities.

    For each label, the label priors are estimated as
        (P + sum_i w_i y_i) / (P + N + sum_i w_i),
    where y_i is the ith label, w_i is the ith weight, P is a pseudo-count of
    positive labels, and N is a pseudo-count of negative labels. The index i
    ranges over all labels observed during all evaluations of the returned op.

    Args:
      labels: A `Tensor` with shape [batch_size, num_labels]. Entries should be
        in [0, 1].
      weights: Coefficients representing the weight of each label. Must be either
        a Tensor of shape [batch_size, num_labels] or `None`, in which case each
        weight is treated as 1.0.
      positive_pseudocount: Number of positive labels used to initialize the label
        priors.
      negative_pseudocount: Number of negative labels used to initialize the label
        priors.
      variables_collections: Optional list of collections for created variables.

    Returns:
      label_priors: An op to update the weighted label_priors. Gives the
        current value of the label priors when evaluated.
    """
    dtype = labels.dtype.base_dtype
    num_labels = get_num_labels(labels)

    if weights is None:
        weights = tf.ones_like(labels)

    # We disable partitioning while constructing dual variables because they will
    # be updated with assign, which is not available for partitioned variables.
    partitioner = tf.get_variable_scope().partitioner
    try:
        tf.get_variable_scope().set_partitioner(None)
        # Create variable and update op for weighted label counts.
        weighted_label_counts = tf.contrib.framework.model_variable(
            name='weighted_label_counts',
            shape=[num_labels],
            dtype=dtype,
            initializer=tf.constant_initializer(
                [positive_pseudocount] * num_labels, dtype=dtype),
            collections=variables_collections,
            trainable=False)
        weighted_label_counts_update = weighted_label_counts.assign_add(
            tf.reduce_sum(weights * labels, 0))

        # Create variable and update op for the sum of the weights.
        weight_sum = tf.contrib.framework.model_variable(
            name='weight_sum',
            shape=[num_labels],
            dtype=dtype,
            initializer=tf.constant_initializer(
                [positive_pseudocount + negative_pseudocount] * num_labels,
                dtype=dtype),
            collections=variables_collections,
            trainable=False)
        weight_sum_update = weight_sum.assign_add(tf.reduce_sum(weights, 0))

    finally:
        tf.get_variable_scope().set_partitioner(partitioner)

    label_priors = tf.div(
        weighted_label_counts_update,
        weight_sum_update)
    return label_priors


def _prepare_labels_logits_weights(labels, logits, weights):
    """Validates labels, logits, and weights.

    Converts inputs to tensors, checks shape compatibility, and casts dtype if
    necessary.

    Args:
      labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
      logits: A `Tensor` with the same shape as `labels`.
      weights: Either `None` or a `Tensor` with shape broadcastable to `logits`.

    Returns:
      labels: Same as `labels` arg after possible conversion to tensor, cast, and
        reshape.
      logits: Same as `logits` arg after possible conversion to tensor and
        reshape.
      weights: Same as `weights` arg after possible conversion, cast, and reshape.
      original_shape: Shape of `labels` and `logits` before reshape.

    Raises:
      ValueError: If `labels` and `logits` do not have the same shape.
    """
    # Convert `labels` and `logits` to Tensors and standardize dtypes.
    logits = tf.convert_to_tensor(logits, name='logits')
    labels = convert_and_cast(labels, 'labels', logits.dtype.base_dtype)
    weights = convert_and_cast(weights, 'weights', logits.dtype.base_dtype)

    try:
        labels.get_shape().merge_with(logits.get_shape())
    except ValueError:
        raise ValueError('logits and labels must have the same shape (%s vs %s)' %
                         (logits.get_shape(), labels.get_shape()))

    original_shape = labels.get_shape().as_list()
    if labels.get_shape().ndims > 0:
        original_shape[0] = -1
    if labels.get_shape().ndims <= 1:
        labels = tf.reshape(labels, [-1, 1])
        logits = tf.reshape(logits, [-1, 1])

    if weights.get_shape().ndims == 1:
        # Weights has shape [batch_size]. Reshape to [batch_size, 1].
        weights = tf.reshape(weights, [-1, 1])
    if weights.get_shape().ndims == 0:
        # Weights is a scalar. Change shape of weights to match logits.
        weights *= tf.ones_like(logits)

    return labels, logits, weights, original_shape


def _range_to_anchors_and_delta(precision_range, num_anchors, dtype):
    """Calculates anchor points from precision range.

    Args:
      precision_range: As required in precision_recall_auc_loss.
      num_anchors: int, number of equally spaced anchor points.
      dtype: Data type of returned tensors.

    Returns:
      precision_values: A `Tensor` of data type dtype with equally spaced values
        in the interval precision_range.
      delta: The spacing between the values in precision_values.

    Raises:
      ValueError: If precision_range is invalid.
    """
    # Validate precision_range.
    if not 0 <= precision_range[0] <= precision_range[-1] <= 1:
        raise ValueError('precision values must obey 0 <= %f <= %f <= 1' %
                         (precision_range[0], precision_range[-1]))
    if not 0 < len(precision_range) < 3:
        raise ValueError('length of precision_range (%d) must be 1 or 2' %
                         len(precision_range))

    # Sets precision_values uniformly between min_precision and max_precision.
    values = np.linspace(start=precision_range[0],
                         stop=precision_range[1],
                         num=num_anchors + 2)[1:-1]
    precision_values = convert_and_cast(
        values, 'precision_values', dtype)
    delta = convert_and_cast(
        values[0] - precision_range[0], 'delta', dtype)
    # Makes precision_values [1, 1, num_anchors].
    precision_values = expand_outer(precision_values, 3)
    return precision_values, delta
