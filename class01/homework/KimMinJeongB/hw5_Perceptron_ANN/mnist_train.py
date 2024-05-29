import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
#fashion_mnist = tk.keras.datasets.fashion_mnist

(image_train, label_train), (image_test, label_test) = mnist.load_data()
#(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()

# normalized images
image_train, image_test = image_train/255.0, image_test/255.0
#f_image_train, f_image_test = f_image_train/255.0, f_image_test/255.0

# fashion_mnist 의 레이블은 숫자로 저장이 되어 있기 때문에 레이블과 클래스 이름을 매핑
#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# plt.figure(figsize=(10,10))
# for i in range(10):
#     plt.subplot(3,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image_train[i])
#     plt.xlabel(label_train[i])
# plt.show()


# ANN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.fit(image_train, label_train, epochs=10, batch_size=10)
model.summary()
model.save('mnist.h5')
