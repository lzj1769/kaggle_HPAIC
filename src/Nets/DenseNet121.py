from keras_applications.densenet import DenseNet121
from keras_applications.densenet import preprocess_input

import keras
from keras import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.layers import BatchNormalization

preprocess_input = preprocess_input


def build_model(input_shape, num_classes, weights='imagenet'):
    # create the base pre-trained model

    base_model = DenseNet121(weights=weights, include_top=False, input_shape=input_shape,
                             backend=keras.backend, layers=keras.layers, models=keras.models,
                             utils=keras.utils)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization(name="batch_1")(x)
    x = Dense(1024, activation='relu', name='fc2014_1')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization(name="batch_2")(x)
    x = Dense(1024, activation='relu', name='fc2014_2')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='sigmoid', name='fc28')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=x, name='pre_trained_resnet50')

    return model
