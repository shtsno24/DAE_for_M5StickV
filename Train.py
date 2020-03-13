import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
device_list = device_lib.list_local_devices()

import Model


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

try:
    with tf.device('/cpu:0'):
        EPOCHS = 100
        TRAIN_DATASET_SIZE = 50000
        TEST_DATASET_SIZE = 10000
        BATCH_SIZE = 100
        # Load dataset
        print("Load dataset...\n\n")
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
        print(train_images.shape, type(train_images))
        print(test_images.shape, type(test_images))
        train_images, test_images = train_images / 255, test_images / 255
        print("\n\nDone")

    # Load model
    print("Load Model...\n\n")
    model = Model.DAE_Net()
    model.summary()
    print("\nDone")

    try:
        # Train model
        print("\n\nTrain Model...")
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
        model.fit(train_images, train_images, validation_data=(test_images, test_images), epochs=EPOCHS,
                  steps_per_epoch=int(TRAIN_DATASET_SIZE / BATCH_SIZE),
                  validation_steps=int(TEST_DATASET_SIZE / BATCH_SIZE))
        print("  Done\n\n")
    except:
        import traceback
        traceback.print_exc()

    try:
        # Save model
        print("\n\nSave Model...")
        model.save('Model.h5')
        print("  Done\n\n")
    except:
        import traceback
        traceback.print_exc()

except:
    import traceback
    traceback.print_exc()

finally:
    input(">")
