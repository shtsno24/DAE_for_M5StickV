import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, Activation, Concatenate, BatchNormalization, Reshape
from tensorflow.keras.layers import LeakyReLU, Add, PReLU, SpatialDropout2D, ZeroPadding2D, Softmax, ReLU
from tensorflow.keras.losses import sparse_categorical_crossentropy


def DAE_Net(input_shape=(32, 32, 3), Momentum=0.1, Dropout_rate=0.01):
    inputs = Input(shape=input_shape)

    x_skip = inputs
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(inputs)
    x = Conv2D(16, (3, 3))(x)
    x = BatchNormalization(momentum=Momentum)(x)
    x = SpatialDropout2D(rate=Dropout_rate)(x)
    x = ReLU()(x)
    x = Concatenate(axis=3)([x, x_skip])
    x_0 = x

    x = MaxPooling2D(pool_size=(2, 2))(x)
    for i in range(2):
        if i < 1:
            x_skip = x
        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = Conv2D(32, (3, 3))(x)
        x = BatchNormalization(momentum=Momentum)(x)
        x = SpatialDropout2D(rate=Dropout_rate)(x)
        x = ReLU()(x)
        if i < 1:
            x = Concatenate(axis=3)([x, x_skip])

    x_1 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)
    for i in range(2):
        if i < 1:
            x_skip = x
        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = Conv2D(32, (3, 3))(x)
        x = BatchNormalization(momentum=Momentum)(x)
        x = SpatialDropout2D(rate=Dropout_rate)(x)
        x = ReLU()(x)
        if i < 1:
            x = Concatenate(axis=3)([x, x_skip])

    x_0 = MaxPooling2D(pool_size=(4, 4))(x_0)
    x_1 = MaxPooling2D(pool_size=(2, 2))(x_1)
    x = Concatenate(axis=3)([x, x_0, x_1])
    x = MaxPooling2D(pool_size=(2, 2))(x)

    for i in range(3):
        x_skip = x
        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = Conv2D(32, (3, 3))(x)
        x = BatchNormalization(momentum=Momentum)(x)
        x = SpatialDropout2D(rate=Dropout_rate)(x)
        x = ReLU()(x)
        x = Concatenate(axis=3)([x, x_skip])

    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = Conv2D(64, (3, 3))(x)
    x = BatchNormalization(momentum=Momentum)(x)
    x = SpatialDropout2D(rate=Dropout_rate)(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x_skip = x
    for _ in range(2):
        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = Conv2D(32, (3, 3))(x)
        x = BatchNormalization(momentum=Momentum)(x)
        x = SpatialDropout2D(rate=Dropout_rate)(x)
        x = ReLU()(x)
    x = Concatenate(axis=3)([x, x_skip])

    x = UpSampling2D(size=(2, 2))(x)
    x_skip = x
    for _ in range(2):
        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = Conv2D(16, (3, 3))(x)
        x = BatchNormalization(momentum=Momentum)(x)
        x = SpatialDropout2D(rate=Dropout_rate)(x)
        x = ReLU()(x)

    x = Concatenate(axis=3)([x, x_skip])

    x = UpSampling2D(size=(2, 2))(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = Conv2D(16, (3, 3))(x)
    x = BatchNormalization(momentum=Momentum)(x)
    x = SpatialDropout2D(rate=Dropout_rate)(x)
    x = ReLU()(x)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = Conv2D(3, (3, 3))(x)
    x = BatchNormalization(momentum=Momentum)(x)
    x = SpatialDropout2D(rate=Dropout_rate)(x)

    outputs = x

    model = Model(inputs, outputs)
    return model
