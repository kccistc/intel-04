import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

mnist = tf.keras.datasets.mnist

(image_train, label_train), (image_test, label_test) = mnist.load_data()
image_train, image_test = image_train/255.0, image_test/255.0

# ANN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(image_train, label_train, validation_data = (image_test, label_test), epochs=10, batch_size=10)
model.summary()
model.save('Mnist_ann.h5')
with open('historyMnist_ann', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)