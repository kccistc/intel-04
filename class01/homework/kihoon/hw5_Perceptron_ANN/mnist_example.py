import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Fashion MNIST 데이터셋 로드
fashion_mnist = tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()

# 데이터 정규화 (0~1 사이의 값으로)
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

# 클래스 이름 정의
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 이미지 시각화
# plt.figure(figsize=(10, 10))
# for i in range(10):
#     plt.subplot(3, 4, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(f_image_train[i])
#     plt.xlabel(class_names[f_label_train[i]])

# plt.show()



model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation= 'relu'))
model.add(tf.keras.layers.Dense(64, activation= 'relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.fit(f_image_train, f_label_train, epochs=10, batch_size=10)

model.summary()
model.save('fashion_mnist.h5')