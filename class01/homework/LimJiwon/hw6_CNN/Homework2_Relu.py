import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import pickle
import cv2

mnist = tf.keras.datasets.mnist
fashion_mnist = tf.keras.datasets.fashion_mnist

(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,10))

for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(class_names[f_label_train[i]])
plt.show()

# MODEL CONSTRUCTION (relu)
# CNN
model = Sequential()
model.add(Conv2D(32, (2,2), activation="relu", input_shape=(28,28,1)))
model.add(Conv2D(64, (2,2), activation="relu"))
model.add(Conv2D(128, (2,2), 2, activation="relu"))
model.add(Conv2D(32, (2,2), activation="relu"))
model.add(Conv2D(64, (2,2), activation="relu"))
model.add(Conv2D(128, (2,2), 2, activation="relu"))

# ANN
model.add(Flatten())
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics=["accuracy"])
history = model.fit(f_image_train, f_label_train, validation_data = (f_image_test, f_label_test), epochs=10, batch_size=10)
model.summary()
model.save('fashion_mnist_Relu.h5')
with open('historyBatchReLu', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

model = tf.keras.models.load_model('./fashion_mnist_Relu.h5')
fashion_mnist = tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()

f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

num = 10
predict = model.predict(f_image_train[:num])
print(f_label_train[:num])
print(" * Prediction, ", np.argmax(predict, axis = 1))