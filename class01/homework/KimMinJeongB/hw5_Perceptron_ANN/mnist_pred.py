import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

model = tf.keras.models.load_model('./mnist.h5')
mnist = tf.keras.datasets.mnist
(image_train, label_train), (image_test, label_test) = mnist.load_data()

image_train, image_test = image_train/255.0, image_test/255.0

num = 10
predict = model.predict(image_test[:num])
print(label_train[:num])
print(" * Prediction, ", np.argmax(predict, axis=1))