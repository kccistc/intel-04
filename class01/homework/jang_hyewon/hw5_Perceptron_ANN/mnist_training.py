import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np

# Model load:MNIST / Fashion MNIST dataset
mnist = tf.keras.datasets.mnist
fashion_mnist = tf.keras.datasets.fashion_mnist

# Model load: MNIST
# 보통 8:2 정도
(image_train, label_train), (image_test, label_test) = mnist.load_data()

# Normalized 
image_train, image_test = image_train / 255.0, image_test / 255.0

# Class name for fashion mnist
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"]

# plt.figure(figsize=(10, 10))
# for i in range(10):
#     plt.subplot(3, 4, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image_train[i])
#     plt.xlabel(label_train[i])
# plt.show()

# ANN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation="relu"))
# model.add(tf.keras.layers.Dense(128, activation="relu"))
# model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()
model.fit(image_train, label_train, epochs=10, batch_size=10)
model.summary()
model.save('mnist.h5')