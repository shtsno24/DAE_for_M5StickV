import tensorflow as tf
import numpy as np
import Model
from PIL import Image

try:
    MODEL_FILE = "Model.h5"
    MODEL_TFLITE = "Model.tflite"
    TEST_IMG = "Img/Test_img.png"

    with tf.device('/cpu:0'):
        # Load model
        print("\n\nLoad Model...\n")
        model = tf.keras.models.load_model(MODEL_FILE)
        model.summary()
        print("\nDone")

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                tf.lite.OpsSet.SELECT_TF_OPS]
        tfmodel = converter.convert() 
        with open(MODEL_TFLITE, "wb") as m:
            m.write(tfmodel)

        interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        test_img = Image.open(TEST_IMG)
        test_data = np.array(test_img, dtype=np.float32)
        test_data_shape = test_data.shape
        test_data /= 255.0
        test_data = test_data.reshape(input_shape)
        interpreter.set_tensor(input_details[0]['index'], test_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data *= 255
        print(output_data.shape, output_data.dtype)
        output_data = output_data.reshape(test_data_shape)
        output_img = Image.fromarray(output_data.astype(np.uint8))
        output_img.save("Img/Prediction_img_tflite.png")

except:
    import traceback
    traceback.print_exc()