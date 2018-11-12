"""ResNet-18 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Slightly modified Felix Yu's (https://github.com/flyyufelix) implementation of
ResNet-101 to have consistent API as those pre-trained models within
`keras.applications`. The original implementation is found here
https://gist.github.com/flyyufelix/65018873f8cb2bbe95f429c474aa1294#file-resnet-101_keras-py

Implementation is based on Keras 2.0
"""
from keras.layers import Input, Dense, AveragePooling2D
from keras.layers import Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
from keras.engine import Layer, InputSpec
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.utils.data_utils import get_file

from keras import backend
from keras import layers

import warnings
import sys

sys.setrecursionlimit(3000)

WEIGHTS_PATH_TF = 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000_no_top.h5'
MD5_HASH_TF = '318e3ac0cd98d51e917526c9f62f0b50'

batch_size = 16
input_shape = (1024, 1024, 3)


class Scale(Layer):
    '''Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:

        out = in * gamma + beta,

    where 'gamma' and 'beta' are the weights and biases larned.

    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights`
            argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights`
            argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights`
            argument.
    '''

    def __init__(self,
                 weights=None,
                 axis=-1,
                 momentum=0.9,
                 beta_init='zero',
                 gamma_init='one',
                 **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(
            self.gamma_init(shape),
            name='{}_gamma'.format(self.name))
        self.beta = K.variable(
            self.beta_init(shape),
            name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(
            self.gamma,
            broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, kernel_size, strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      kernel_initializer='he_normal',
                      padding='same',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    shortcut = layers.Conv2D(filters2, kernel_size, strides=strides,
                             padding='same',
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet18(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the ResNet-101 architecture.

    Optionally loads weights pre-trained on ImageNet. Note that when using
    TensorFlow, for best performance you should set
    image_data_format='channels_last'` in your Keras config at
    ~/.keras/keras.json.

    The model and the weights are compatible with both TensorFlow and Theano.
    The data format convention used by the model is the one specified in your
    Keras config file.

    Parameters
    ----------
        include_top: whether to include the fully-connected layer at the top of
            the network.
        weights: one of `None` (random initialization) or 'imagenet'
            (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified if
            `include_top` is False (otherwise the input shape has to be
            `(224, 224, 3)` (with `channels_last` data format) or
            `(3, 224, 224)` (with `channels_first` data format). It should have
            exactly 3 inputs channels, and width and height should be no
            smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction when
            `include_top` is `False`.
            - `None` means that the output of the model will be the 4D tensor
                output of the last convolutional layer.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional layer, and thus the output of
                the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True, and if no `weights` argument is
            specified.

    Returns
    -------
        A Keras model instance.

    Raises
    ------
        ValueError: in case of invalid argument for `weights`, or invalid input
        shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='data')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(
                tensor=input_tensor, shape=input_shape, name='data')
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64], stage=2, block='b')

    x = conv_block(x, 3, [128, 128], stage=3, block='a')
    x = identity_block(x, 3, [128, 128], stage=3, block='b')

    x = conv_block(x, 3, [256, 256], stage=4, block='a')
    x = identity_block(x, 3, [256, 256], stage=4, block='b')

    x = conv_block(x, 3, [512, 512], stage=5, block='a')
    x = identity_block(x, 3, [512, 512], stage=5, block='b')

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet101')

    # load weights
    if weights == 'imagenet':
        filename = 'resnet101_weights_{}.h5'.format(K.image_dim_ordering())
        path = WEIGHTS_PATH_TF
        md5_hash = MD5_HASH_TF
        weights_path = get_file(
            fname=filename,
            origin=path,
            cache_subdir='models',
            md5_hash=md5_hash,
            hash_algorithm='md5')
        model.load_weights(weights_path, by_name=True)

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
    return model


def build_model(num_classes, weights='imagenet'):
    # create the base pre-trained model

    base_model = ResNet18(weights=weights,
                          include_top=False,
                          input_shape=input_shape)

    # add a global spatial average pooling layer
    x = base_model.output
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = BatchNormalization(name="batch_1")(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', name='fc1024_1')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization(name="batch_2")(x)
    x = Dense(1024, activation='relu', name='fc1024_2')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='sigmoid', name='fc28')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=x, name='resnet101')

    return model


# from keras.losses import binary_crossentropy
#
# model = build_model(num_classes=28)
# model.compile(optimizer="adam", loss=binary_crossentropy)
# model.summary()
