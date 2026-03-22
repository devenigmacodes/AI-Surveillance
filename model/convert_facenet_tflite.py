import tensorflow as tf
from keras_facenet import FaceNet

embedder = FaceNet()
model = embedder.model


model.save("model/facenet_model.h5")
print("Saved FaceNet model as facenet_model.h5")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()


with open("model/facenet_model_quantized.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("Quantized model saved as facenet_model_quantized.tflite")