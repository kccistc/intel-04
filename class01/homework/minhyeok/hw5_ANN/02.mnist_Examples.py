import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

'''
주석으로 되어 있는 부분은 fashion mnist에 관한 코드입니다.
현재는 mnist만 되어 있는 부분만 남겨 놓았습니다.
'''


# Model load: MNIST / Fashion MNIST Dataset
mnist = tf.keras.datasets.mnist
# fashion_mnist = tf.keras.datasets.fashion_mnist

# (f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
# # normalized iamges
# f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

(image_train, label_train), (image_test, label_test) = mnist.load_data()
# normalized iamges
image_train, image_test = image_train / 255.0, image_test / 255.0

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
# 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure(figsize=(10,10))
# for i in range(10):
#     plt.subplot(3,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(f_image_train[i])
#     plt.xlabel(class_names[f_label_train[i]])
# plt.show(block = False)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
# model.fit(f_image_train, f_label_train, epochs=10, batch_size=10)
# model.summary()
# model.save('fashion_mnist.h5')

model.fit(image_train, label_train, epochs=10, batch_size=10)
model.summary()
model.save('mnist.h5')