import tensorflow as tf
import numpy as np
import Model_V0_1 as Model

try:
    # MODEL_FILE = "TestNet_VOC2012_npz.h5"
    # MODEL_TFLITE = "TestNet_VOC2012_npz.tflite"
    # MODEL_FILE = "model_viewer.h5"
    # MODEL_TFLITE = "model_viewer.tflite"
    MODEL_FILE = "Model_V0_1.h5"
    MODEL_TFLITE = "Model_V0_1.tflite"
    LABELS = 21
    COLOR_DEPTH = 3
    CROP_HEIGHT = 128  # sensor.LCD[128, 160]
    CROP_WIDTH = 160

    with tf.device('/cpu:0'):
        # Load model
        print("\n\nLoad Model...\n")
        model = tf.keras.models.load_model(MODEL_FILE, custom_objects={'loss_function': Model.weighted_SparseCategoricalCrossentropy(LABELS)})
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
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)

except:
    import traceback
    traceback.print_exc()