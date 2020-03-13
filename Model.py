import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, Activation, Concatenate, BatchNormalization, Reshape
from tensorflow.keras.layers import LeakyReLU, Add, PReLU, SpatialDropout2D, ZeroPadding2D, Softmax, ReLU
from tensorflow.keras.losses import sparse_categorical_crossentropy


def Multi_Conv(x, output_depth, size="same", activation=True, internal_depth_ratio=4, Momentum=0.1, Dropout_rate=0.01):

    internal_depth = int(output_depth / internal_depth_ratio)

    x_skip = x
    x_skip = MaxPooling2D(pool_size=(2, 2))(x_skip)
    x_skip = UpSampling2D(size=(2, 2))(x_skip)

    x_conv = Conv2D(internal_depth, (1, 1))(x)
    x_conv = BatchNormalization(momentum=Momentum)(x_conv)
    x_conv = SpatialDropout2D(rate=Dropout_rate)(x_conv)
    x_conv = ReLU()(x_conv)

    x_conv = ZeroPadding2D(padding=((1, 1), (1, 1)))(x_conv)
    x_conv = DepthwiseConv2D((3, 3))(x_conv)
    x_conv = BatchNormalization(momentum=Momentum)(x_conv)
    x_conv = ReLU()(x_conv)

    x_conv = Conv2D(internal_depth, (1, 1))(x_conv)
    x_conv = BatchNormalization(momentum=Momentum)(x_conv)
    x_conv = SpatialDropout2D(rate=Dropout_rate)(x_conv)
    x_conv = ReLU()(x_conv)

    x = Concatenate(axis=3)([x_skip, x_conv])

    if size == "up":
        x = UpSampling2D(size=(2, 2))(x)
    if size == "down":
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = Conv2D(output_depth, (3, 3))(x)
    if activation is True:
        x = BatchNormalization(momentum=Momentum)(x)
        x = SpatialDropout2D(rate=Dropout_rate)(x)
        x = ReLU()(x)

    return x


def DAE_Net(input_shape=(32, 32, 3)):
    inputs = Input(shape=input_shape)

    x_skip = MaxPooling2D(pool_size=(2, 2))(inputs)
    x = Multi_Conv(inputs, 16, internal_depth_ratio=2, size="down")
    x = Multi_Conv(x, 16, internal_depth_ratio=2)
    x = Concatenate(axis=3)([x, x_skip])
    x_0 = x

    x_skip = MaxPooling2D(pool_size=(2, 2))(x)
    x = Multi_Conv(x_0, 32, size="down")
    x = Multi_Conv(x, 32)
    x = Concatenate(axis=3)([x, x_skip])
    x_1 = x

    x_skip = MaxPooling2D(pool_size=(2, 2))(x)
    x = Multi_Conv(x, 64, size="down")
    x = Multi_Conv(x, 64)
    x = Concatenate(axis=3)([x, x_skip])
    x_2 = x

    x_1 = UpSampling2D(size=(2, 2))(x_1)
    x_2 = UpSampling2D(size=(4, 4))(x_2)
    x = Concatenate(axis=3)([x_0, x_1, x_2])
    x = Multi_Conv(x, 8, internal_depth_ratio=1)

    x_skip = x
    x = Multi_Conv(x, 16)
    x = Multi_Conv(x, 16, size="up")
    x_skip = UpSampling2D(size=(2, 2))(x_skip)
    x = Concatenate(axis=3)([x, x_skip])

    x_skip = x
    x = Multi_Conv(x, 8, internal_depth_ratio=2)
    x = Multi_Conv(x, 8, internal_depth_ratio=2)
    x = Concatenate(axis=3)([x, x_skip])
    x = Multi_Conv(x, 3, internal_depth_ratio=1)
    x = Multi_Conv(x, 3, internal_depth_ratio=1, activation=False)
    x = ReLU()(x)
    outputs = x

    model = Model(inputs, outputs)
    return model


def DAE_Net_2(input_shape=(32, 32, 3), Momentum=0.1, Dropout_rate=0.01):
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
