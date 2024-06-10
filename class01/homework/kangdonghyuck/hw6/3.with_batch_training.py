import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle




mnist = tf.keras.datasets.mnist

(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()

f_image_train, f_image_test = f_image_train/255.0, f_image_test/255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']




plt.figure(figsize =(10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel([f_label_train[i]])
plt.show()



model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32,(2,2),activation="sigmoid", input_shape=(28,28,1)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64,(2,2),activation="sigmoid"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128,(2,2),2,activation="sigmoid"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(32,(2,2),activation="sigmoid"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64,(2,2),activation="sigmoid"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128,(2,2),2,activation="sigmoid"))
model.add(tf.keras.layers.BatchNormalization())


#########################################################

# ANN
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))




model.compile(
        loss = "sparse_categorical_crossentropy",
        optimizer = "adam",
        metrics=["accuracy"])
history = model.fit(f_image_train, f_label_train, validation_data=(f_image_test, f_label_test),epochs=10, batch_size=256)
model.summary()
model.save('fashion_mnist_withbatch.h5')
with open('historywith_batch', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
    
    
    