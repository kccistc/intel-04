# Load model
# tf.keras.models.load_model()
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

mnist = tf.keras.datasets.mnist # 글자 데이터 셋
fashion_mnist = tf.keras.datasets.fashion_mnist # 의류 데이터 셋

(f_image_train,f_label_train),(f_image_test,f_label_test) = fashion_mnist.load_data()
# 훈련 데이터셋 과 테스트 데이터셋을 나눈다.
f_image_train, f_image_test = f_image_train/ 255.0,f_image_test/ 255.0
# image의 값을 정규화한 값으로 나타낸다.(0~1값으로 표현)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(image_train,label_train),(image_test,label_test) = mnist.load_data()
# 훈련 데이터셋 과 테스트 데이터셋을 나눈다.
image_train, image_test = image_train/ 255.0,image_test/ 255.0


model = tf.keras.models.load_model('./mnist_ch.h5')
mnist = tf.keras.datasets.mnist
(image_train, label_train), (image_test, label_test) = mnist.load_data()
image_train, image_test = image_train / 255.0, image_test / 255.0
num = 10
predict = model.predict(image_test[:num])
print(label_train[:num])
print(" * Prediction, ", np.argmax(predict, axis = 1))