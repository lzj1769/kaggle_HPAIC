from keras import Model
from keras import layers
from keras.losses import binary_crossentropy


def residual_block(input_tensor, input_channels=None, output_channels=None,
                   kernel_size=(3, 3), stride=1):
    """
    full pre-activation residual block
    https://arxiv.org/pdf/1603.05027.pdf
    """
    if output_channels is None:
        output_channels = input_tensor.shape[-1]
    if input_channels is None:
        input_channels = output_channels // 4

    strides = (stride, stride)

    x = layers.BatchNormalization()(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(input_channels, (1, 1))(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(input_channels, kernel_size, padding='same', strides=stride)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(output_channels, (1, 1), padding='same')(x)

    if input_channels != output_channels or stride != 1:
        input_tensor = layers.Conv2D(output_channels, (1, 1), padding='same', strides=strides)(input)

    x = layers.add([x, input_tensor])
    return x


def attention_block(input, input_channels=None, output_channels=None, encoder_depth=1):
    """
    attention block
    https://arxiv.org/abs/1704.06904
    """

    p = 1
    t = 2
    r = 1

    if input_channels is None:
        input_channels = input.get_shape()[-1].value
    if output_channels is None:
        output_channels = input_channels

    # First Residual Block
    input = residual_block(input)

    # Trunc Branch
    output_trunk = input
    for i in range(t):
        output_trunk = residual_block(output_trunk)

    # Soft Mask Branch
    ## encoder
    ### first down sampling
    output_soft_mask = layers.MaxPool2D(padding='same')(input)  # 32x32
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


def AttentionResNet56(shape=(224, 224, 3), n_channels=64, n_classes=100):
    """
    Attention-92 ResNet
    https://arxiv.org/abs/1704.06904
    """
    img_input = layers.Input(shape=shape)

    x = layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', padding='same', use_bias=False)(img_input)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool1', padding='same')(x)

    x = residual_block(x, output_channels=n_channels * 4)  # 56x56
    x = attention_block(x, encoder_depth=3)  # bottleneck 7x7
    x = residual_block(x, output_channels=n_channels * 32, stride=2)  # 7x7

    x = layers.GlobalAvgPool2D()(x)
    x = layers.BatchNormalization(name="batch_1")(x)
    x = layers.Dense(1024, activation='relu', name='fc1024_1')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization(name="batch_2")(x)
    x = layers.Dense(1024, activation='relu', name='fc1024_2')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(n_classes, activation='sigmoid', name='fc28')(x)

    return Model(img_input, output)


model = AttentionResNet56(shape=(512, 512, 4), n_classes=28)
model.compile(optimizer="adam", loss=binary_crossentropy)
model.summary()
