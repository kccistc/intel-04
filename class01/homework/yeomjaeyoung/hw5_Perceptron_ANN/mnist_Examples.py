import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist # 글자 데이터 셋
fashion_mnist = tf.keras.datasets.fashion_mnist # 의류 데이터 셋

(f_image_train,f_label_train),(f_image_test,f_label_test) = fashion_mnist.load_data()
# 훈련 데이터셋 과 테스트 데이터셋을 나눈다.
f_image_train, f_image_test = f_image_train/ 255.0,f_image_test/ 255.0
# image의 값을 정규화한 값으로 나타낸다.(0~1값으로 표현)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize = (10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(class_names[f_label_train[i]])
plt.show(block = True)

(image_train,label_train),(image_test,label_test) = mnist.load_data()
# 훈련 데이터셋 과 테스트 데이터셋을 나눈다.
image_train, image_test = image_train/ 255.0,image_test/ 255.0

plt.figure(figsize = (10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_train[i])
    plt.xlabel([label_train[i]])
plt.show(block = True)
