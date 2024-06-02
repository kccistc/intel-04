import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# 모델 불러오기
model = tf.keras.models.load_model('./fashion_mnist.h5')
fashion_mnist = tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()

# 이미지 정규화
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

# num개의 이미지를 예측
num = 10
predict = model.predict(f_image_test[:num])
print(" * Actual, ", f_label_test[:num])
print(" * Prediction, ", np.argmax(predict, axis = 1))
