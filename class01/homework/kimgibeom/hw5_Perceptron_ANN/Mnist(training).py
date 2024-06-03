import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()
f_image_train, f_image_test = f_image_train/255.0, f_image_test/255.0

# ANN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(f_image_train, f_label_train, epochs=10, batch_size=10)
model.summary()
model.save('mnist.h5')