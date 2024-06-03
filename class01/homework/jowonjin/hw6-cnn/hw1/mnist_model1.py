import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Model load: MNIST / Fashion MNIST Dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
mnist = tf.keras.datasets.mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
# Normalized images
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

(m_image_train, m_label_train), (m_image_test, m_label_test) = mnist.load_data()
# Normalized images
m_image_train, m_image_test = m_image_train / 255.0, m_image_test / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(class_names[f_label_train[i]])
plt.show()

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(m_image_train[i])
plt.show()

# ANN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32,(2,2),activation="sigmoid",input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(64,(2,2),activation="sigmoid"))
model.add(tf.keras.layers.Conv2D(128,(2,2),2,activation="sigmoid"))
model.add(tf.keras.layers.Conv2D(32,(2,2),activation="sigmoid"))
model.add(tf.keras.layers.Conv2D(64,(2,2),activation="sigmoid"))
model.add(tf.keras.layers.Conv2D(128,(2,2),2,activation="sigmoid"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history= model.fit(f_image_train, f_label_train, validation_data=(f_image_train, f_label_train,),epochs=10, batch_size=256)
model.summary()
model.save('fashion_mnist.h5')
with open('historynoBatch', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

model = tf.keras.models.load_model('./fashion_mnist.h5')
fashion_mnist = tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()

f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

num = 10
predict = model.predict(f_image_test[:num])
print(f_label_train[:num])
print(" * Prediction, ", np.argmax(predict, axis = 1))

import pickle
import numpy as np
import matplotlib.pyplot as plt

historynoBatch = pickle.load(open('./historynoBatch', "rb"))

val_accNB = historynoBatch["val_accuracy"]
val_lossNB= historynoBatch["val_loss"]









