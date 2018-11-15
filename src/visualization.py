from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

sns.set(rc={'figure.figsize': (12, 10)})

from configure import *


def visua_acc_loss(acc_loss_path, logs_path, exp_config):
    # load logs
    loss_filename = os.path.join(logs_path, "{}.log".format(exp_config))
    df = pd.read_csv(loss_filename)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.plot(df['binary_accuracy'])
    ax1.plot(df['val_binary_accuracy'])
    ax1.set_title('Accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left')

    ax2.plot(df['loss'])
    ax2.plot(df['val_loss'])
    ax2.set_title('Loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')

    fig.tight_layout()
    fig.savefig(os.path.join(acc_loss_path, "{}.pdf".format(exp_config)))


def visua_threshold_f1(f1_score, optimal_thresholds, exp_id):
    thresholds = np.linspace(0, 1, 100)
    optimal_thresholds = np.round(optimal_thresholds, 2)

    plt.style.use("default")
    fig, ax = plt.subplots(7, 4, figsize=(10, 12))
    for i in range(7):
        for j in range(4):
            if i * 4 + j < 28:
                ax[i][j].plot(thresholds, f1_score[i * 4 + j])
                ax[i][j].set_title("Class: {}, Threshold: {}".format(i * 4 + j, optimal_thresholds[i * 4 + j]))
                ax[i][j].set_ylim([0, 1])
            else:
                break

    fig.tight_layout()
    fig.savefig(os.path.join(VISUALIZATION_PATH, "{}_threshold_f1.pdf".format(exp_id)))


def visua_f1_classes(f1_score, exp_id):
    fig, ax = plt.subplots(figsize=(8, 8))
    ind = np.arange(N_LABELS)

    plt.bar(ind, f1_score)
    ax.set_xticks(ind)
    xlabels = []
    for i in np.arange(N_LABELS):
        xlabels.append("Class {}".format(i))

    ax.set_xticklabels(ind)
    ax.set_ylim([0, 1])
    ax.set_ylabel('F1 score')
    ax.set_title('Macro F1 score: {}'.format(np.mean(f1_score)))

    fig.tight_layout()
    fig.savefig(os.path.join(VISUALIZATION_PATH, "{}_f1_class.pdf".format(exp_id)))


def apply_modifications(model, custom_objects):
    from keras.models import load_model
    """Applies modifications to the model layers to create a new Graph. For example, simply changing
    `model.layers[idx].activation = new activation` does not change the graph. The entire graph needs to be updated
    with modified inbound and outbound tensors because of change in layer building function.

    Args:
        model: The `keras.models.Model` instance.

    Returns:
        The modified model with changes applied. Does not mutate the original `model`.
    """
    # The strategy is to save the modified model and load it back. This is done because setting the activation
    # in a Keras layer doesnt actually change the graph. We have to iterate the entire graph and change the
    # layer inbound and outbound nodes with modified tensors. This is doubly complicated in Keras 2.x since
    # multiple inbound and outbound nodes are allowed with the Graph API.
    model_path = '/tmp/' + next(tempfile._get_candidate_names()) + '.h5'
    try:
        model.save(model_path)
        return load_model(model_path, custom_objects=custom_objects)
    finally:
        os.remove(model_path)


def visua_cnn(model, custom_objects=None, image=None, id=id):
    from vis.utils.utils import find_layer_idx
    from keras import activations
    from vis.visualization import overlay, visualize_cam

    # Utility to search for layer index by name.
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = find_layer_idx(model, 'fc28')

    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = apply_modifications(model, custom_objects=custom_objects)

    grads = visualize_cam(model, layer_idx, filter_indices=None,
                          seed_input=image, backprop_modifier='guided')

    jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)

    fig = plt.figure(dpi=300, tight_layout=True)
    fig.figimage(jet_heatmap)
    fig.figimage(image, xo=512)

    DPI = fig.get_dpi()
    fig.set_size_inches(2 * 512.0 / float(DPI), 512.0 / float(DPI))
    fig.savefig("{}.png".format(id))


def visua_decode(model, image, id):
    import keras.backend as K
    from vis.utils.utils import find_layer_idx, apply_modifications
    from keras import activations
    from vis.visualization import visualize_cam

    x = np.expand_dims(image.astype(K.floatx()) / 255.0, axis=0)
    [class_output, image_output] = model.predict(x=x)

    # Utility to search for layer index by name.
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = find_layer_idx(model, 'classification')
    model.layers[layer_idx].activation = activations.linear
    model = apply_modifications(model)

    penultimate_layer_idx = find_layer_idx(model, 'res5c_branch2c')

    grads = visualize_cam(model, layer_idx, filter_indices=None,
                          seed_input=x[0], backprop_modifier='guided',
                          penultimate_layer_idx=penultimate_layer_idx)

    jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
    image_output = np.uint8(image_output * 255)

    fig = plt.figure(dpi=300, tight_layout=True)
    fig.figimage(jet_heatmap)
    fig.figimage(image_output[0], xo=512)
    fig.figimage(image, xo=1024)

    DPI = fig.get_dpi()
    fig.set_size_inches(3 * 512.0 / float(DPI), 512.0 / float(DPI))
    fig.savefig("{}.png".format(id))
    print(class_output)
