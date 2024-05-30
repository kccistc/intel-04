import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

fashion_mnist =tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover','Dress','Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

height, width = f_image_train[0].shape[0], f_image_train[0].shape[1]
print(height, width)

inputs = tf.keras.Input(shape=(height,width,1))
rescale = tf.keras.layers.Rescaling(1./255)(inputs)
x = rescale
_x = tf.keras.layers.Conv2D(128,3,activation='relu',padding='same')(x)
_x = tf.keras.layers.BatchNormalization()(_x)
_x = tf.keras.layers.Conv2D(128,3,activation='relu',padding='same')(_x)
_x = tf.keras.layers.BatchNormalization()(_x)
x = _x
_x = tf.keras.layers.Conv2D(128,3,activation='relu',padding='same')(x)
_x = tf.keras.layers.BatchNormalization()(_x)
_x = tf.keras.layers.Conv2D(128,3,activation='relu',padding='same')(_x)
_x = tf.keras.layers.BatchNormalization()(_x)
x = x + _x
x = tf.keras.layers.MaxPooling2D(2)(x)
_x = tf.keras.layers.Conv2D(128,3,activation='relu',padding='same')(x)
_x = tf.keras.layers.BatchNormalization()(_x)
_x = tf.keras.layers.Conv2D(128,3,activation='relu',padding='same')(_x)
_x = tf.keras.layers.BatchNormalization()(_x)
x = x + _x
x = tf.keras.layers.MaxPooling2D(2)(x)
_x = tf.keras.layers.Conv2D(128,3,activation='relu',padding='same')(x)
_x = tf.keras.layers.BatchNormalization()(_x)
_x = tf.keras.layers.Conv2D(128,3,activation='relu',padding='same')(_x)
_x = tf.keras.layers.BatchNormalization()(_x)
x = x + _x
x = tf.keras.layers.MaxPooling2D(2)(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation = 'relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation = 'relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(10, activation='softmax')(x)
outputs = x

model_res = tf.keras.Model(inputs, outputs)
model_res.summary()

model_res.compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics=["accuracy"])
    
history_res = model_res.fit(f_image_train, 
                                    f_label_train, 
                                    validation_data=(f_image_test, f_label_test), 
                                    epochs=10, 
                                    batch_size=256)
model_res.save('fashion_mnist_res.h5')
with open('historyRes', 'wb') as file_pi:
    pickle.dump(history_res.history, file_pi)
