import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

fashion_mnist =tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover','Dress','Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(class_names[f_label_train[i]])
plt.show()

height, width = f_image_train[0].shape[0], f_image_train[0].shape[1]
print(height, width)

# model without batch
model_noBatch = tf.keras.Sequential()
model_noBatch.add(tf.keras.layers.InputLayer(input_shape=(height,width,1)))
model_noBatch.add(tf.keras.layers.Rescaling(1./255))
model_noBatch.add(tf.keras.layers.Conv2D(32,(2,2),activation = "sigmoid"))
model_noBatch.add(tf.keras.layers.Conv2D(64,(2,2),activation = "sigmoid"))
model_noBatch.add(tf.keras.layers.Conv2D(128,(2,2),2,activation = "sigmoid"))
model_noBatch.add(tf.keras.layers.Conv2D(32,(2,2),activation = "sigmoid"))
model_noBatch.add(tf.keras.layers.Conv2D(64,(2,2),activation = "sigmoid"))
model_noBatch.add(tf.keras.layers.Conv2D(128,(2,2),2,activation = "sigmoid"))

model_noBatch.add(tf.keras.layers.Flatten())
model_noBatch.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model_noBatch.add(tf.keras.layers.Dense(10, activation='softmax'))

print(model_noBatch.summary())

model_noBatch.compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics=["accuracy"])

history_noBatch = model_noBatch.fit(f_image_train, 
                                    f_label_train, 
                                    validation_data=(f_image_test, f_label_test), 
                                    epochs=10, 
                                    batch_size=256)
model_noBatch.save('fashion_mnist_noBatch.h5')
with open('historyNoBatch', 'wb') as file_pi:
    pickle.dump(history_noBatch.history, file_pi)
