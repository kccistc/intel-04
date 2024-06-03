import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()
f_image_train, f_image_test = f_image_train/255.0, f_image_test/255.0

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel([f_label_train[i]])
plt.show(block = True)