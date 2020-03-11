import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, Activation, Concatenate, BatchNormalization, Reshape
from tensorflow.keras.layers import LeakyReLU, Add, PReLU, SpatialDropout2D, ZeroPadding2D, Softmax, ReLU
from tensorflow.keras.losses import sparse_categorical_crossentropy


def DAE_Net(input_shape=(32, 32, 3), Momentum=0.1, Dropout_rate=0.01):
    inputs = Input(shape=input_shape)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(inputs)
    x = Conv2D(8, (3, 3))(x)
    x = BatchNormalization(momentum=Momentum)(x)
    x = SpatialDropout2D(rate=Dropout_rate)(x)
    x = ReLU()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    for _ in range(2):
        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = Conv2D(16, (3, 3))(x)
        x = BatchNormalization(momentum=Momentum)(x)
        x = SpatialDropout2D(rate=Dropout_rate)(x)
        x = ReLU()(x)

    x_1 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)
    for _ in range(2):
        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = Conv2D(32, (3, 3))(x)
        x = BatchNormalization(momentum=Momentum)(x)
        x = SpatialDropout2D(rate=Dropout_rate)(x)
        x = ReLU()(x)

    x_2 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)
    for _ in range(4):
        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = Conv2D(32, (3, 3))(x)
        x = BatchNormalization(momentum=Momentum)(x)
        x = SpatialDropout2D(rate=Dropout_rate)(x)
        x = ReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate(axis=3)([x, x_2])
    for _ in range(2):
        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = Conv2D(32, (3, 3))(x)
        x = BatchNormalization(momentum=Momentum)(x)
        x = SpatialDropout2D(rate=Dropout_rate)(x)
        x = ReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate(axis=3)([x, x_1])
    for _ in range(2):
        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = Conv2D(16, (3, 3))(x)
        x = BatchNormalization(momentum=Momentum)(x)
        x = SpatialDropout2D(rate=Dropout_rate)(x)
        x = ReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = Conv2D(3, (3, 3))(x)
    x = BatchNormalization(momentum=Momentum)(x)
    x = SpatialDropout2D(rate=Dropout_rate)(x)

    outputs = Softmax()(x)

    model = Model(inputs, outputs)
    return model
