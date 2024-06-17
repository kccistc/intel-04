import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import os
from PIL import Image 
import tensorflow as tf
import pickle

fashion_mnist= tf.keras.datasets.fashion_mnist
(f_image_train,f_label_train),(f_image_test,f_label_test)=fashion_mnist.load_data()
f_image_train,f_image_test=f_image_train/255.0,f_image_test/255.0
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(class_names[f_label_train[i]])
plt.show()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32,(2,2),activation="sigmoid",
input_shape=(28,28,1)))
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
history = model.fit(f_image_train,f_label_train, validation_data = (f_image_test, f_label_test), epochs=10, batch_size=100)
model.summary()
model.save('historyNoBatch.h5')
with open('historyNoBatch', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

model = tf.keras.models.load_model('./historyNoBatch.h5')
fashion_mnist = tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0
num = 10
predict = model.predict(f_image_test[:num])
print(f_label_train[:num])
print(" * Prediction, ", np.argmax(predict, axis = 1))

