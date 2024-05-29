import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model=tf.keras.models.load_model('./fashion_mnist.h5')
fashion_mnist=tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()

f_image_train, f_image_test = f_image_train/255.0 , f_image_test/255.0

num =10
predict = model.predict(f_image_test[:num])
print(f_label_train[:num])
print(" * Prediction, ", np.argmax(predict, axis=1)) # 

