import numpy as np
import tensorflow as tf
from PIL import Image


try:
    MODEL_FILE = "Model.h5"
    # Load dataset
    print("Load dataset...\n\n")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    print(train_images.shape, type(train_images))
    print(test_images.shape, type(test_images))
    idx = np.random.choice(10000, 1, replace=False)
    test_img = test_images[idx]
    print(test_img.shape, idx)
    print("\n\nDone")

    # Load model
    print("\n\nLoad Model...\n")
    model = tf.keras.models.load_model(MODEL_FILE)
    model.summary()
    print("\nDone")

    # Prediction
    print("\n\nPrediction...\n")
    prediction_img = model.predict(test_img / 255)
    prediction_img = tf.reshape(prediction_img, [32, 32, 3])
    prediction_img = tf.cast(prediction_img, tf.float32)
    prediction_img *= 255
    print(prediction_img.shape, "\nDone")

    # Show Prediction
    print("\n\nSave Image...\n")
    prediction_object = Image.fromarray(prediction_img.numpy().astype(np.uint8))
    prediction_object.save("Img/Prediction_img.png")
    test_object = Image.fromarray(test_img.reshape((32, 32, 3)).astype(np.uint8))
    test_object.save("Img/Test_img.png")
    print("\nDone")

except:
    import traceback
    traceback.print_exc()

finally:
    input(">")
