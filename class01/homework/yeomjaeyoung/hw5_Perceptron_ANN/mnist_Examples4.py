import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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


model = tf.keras.Sequential()
model.add(tf.keras.Input(shape = (28,28)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(image_train, label_train, epochs=10, batch_size=10)
model.summary()
model.save('mnist_ch.h5')