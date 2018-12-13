import sys

from keras import Model
from keras import layers
from keras import backend

sys.setrecursionlimit(3000)

BATCH_SIZE = 16
INPUT_SHAPE = (1024, 1024, 4)
MAX_QUEUE_SIZE = 32
LEARNING_RATE = 1e-04


def residual_block(input_tensor, filters=None,
                   kernel_size=(3, 3)):
    """
    full pre-activation residual block
    https://arxiv.org/pdf/1603.05027.pdf
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


def attention_block(input_tensor, filters=None, encoder_depth=1):
    """
    attention block
    https://arxiv.org/abs/1704.06904
    """

    p = 1
    t = 2
    r = 1

    # First Residual Block
    input_tensor = residual_block(input_tensor, filters)

    # Trunc Branch
    output_trunk = input_tensor
    for i in range(t):
        output_trunk = residual_block(output_trunk)

    # Soft Mask Branch
    ## encoder
    ### first down sampling
    output_soft_mask = layers.MaxPool2D(padding='same')(input_tensor)  # 32x32
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)

    skip_connections = []
    for i in range(encoder_depth - 1):

        ## skip connections
        output_skip_connection = residual_block(output_soft_mask)
        skip_connections.append(output_skip_connection)
        # print ('skip shape:', output_skip_connection.get_shape())

        ## down sampling
        output_soft_mask = layers.MaxPool2D(padding='same')(output_soft_mask)
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)

            # decoder
    skip_connections = list(reversed(skip_connections))
    for i in range(encoder_depth - 1):
        # upsampling
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)
        output_soft_mask = layers.UpSampling2D()(output_soft_mask)
        # skip connections
        output_soft_mask = layers.add([output_soft_mask, skip_connections[i]])

    # last upsampling
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)
    output_soft_mask = layers.UpSampling2D()(output_soft_mask)

    # Output
    output_soft_mask = layers.Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = layers.Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = layers.Activation('sigmoid')(output_soft_mask)

    # Attention: (1 + output_soft_mask) * output_trunk
    output = layers.Lambda(lambda x: x + 1)(output_soft_mask)
    output = layers.multiply([output, output_trunk])

    # Last Residual Block
    for i in range(p):
        output = residual_block(output)

    return output


def AttentionResNet(shape=INPUT_SHAPE, n_channels=64, n_classes=100):
    """
    Attention-92 ResNet
    https://arxiv.org/abs/1704.06904
    """
    img_input = layers.Input(shape=shape)

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=-1, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = residual_block(x, filters=128)  # 56x56
    x = attention_block(x, encoder_depth=3)  # bottleneck 7x7
    x = residual_block(x, filters=256, stride=2)  # 7x7

    x = layers.GlobalAvgPool2D()(x)

    x = layers.BatchNormalization(name="batch_1")(x)
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization(name="batch_2")(x)
    x = layers.Dense(512, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.5)(x)

    output = layers.Dense(n_classes, activation='sigmoid', name='fc28')(x)

    return Model(img_input, output)


model = AttentionResNet(n_classes=28)
print model.summary()
