# import tensorflow as tf

# # Load the Keras model
# model = tf.keras.models.load_model("assets/face_detection_model.h5")

# # Convert the model to TensorFlow Lite format
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# # Save the converted model
# with open("assets/model2.tflite", "wb") as f:
#     f.write(tflite_model)

# print("===================================================")
# print("Model converted successfully to model2.tflite")
# print("===================================================")

# interpreter = tf.lite.Interpreter(model_path="assets/model2.tflite")
# interpreter.allocate_tensors()
# print("===================================================")    
# print("TFLite model loaded successfully!")
# print("===================================================")
