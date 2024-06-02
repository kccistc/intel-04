import pickle

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization


# Fashion MNIST 데이터 셋 불러오기
fashion_mnist = tf.keras.datasets.fashion_mnist

(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0  # Normalize

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 데이터 셋 출력
# plt.figure(figsize=(10,10))
# for i in range(10):
#     plt.subplot(3,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(f_image_train[i])
#     plt.xlabel(class_names[f_label_train[i]])
# plt.show()

set_model = 0           # 모델 구성 선택

# 모델 구성
if set_model == 1:      # With BatchNormalization
    model_name = 'fashion_mnist_batch.h5'
    history_name = 'historyWithBatch'
    model = Sequential()
    model.add(Conv2D(32, (2, 2), activation="sigmoid", input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (2, 2), activation="sigmoid"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (2, 2), 2, activation="sigmoid"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (2, 2), activation="sigmoid"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (2, 2), activation="sigmoid"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (2, 2), 2, activation="sigmoid"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))

elif set_model == 2:    # ReLU
    model_name = 'fashion_mnist_relu.h5'
    history_name = 'historyWithReLU'
    model = Sequential()
    model.add(Conv2D(32, (2, 2), activation="relu", input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (2, 2), activation="relu"))
    model.add(Conv2D(128, (2, 2), 2, activation="relu"))
    model.add(Conv2D(32, (2, 2), activation="relu"))
    model.add(Conv2D(64, (2, 2), activation="relu"))
    model.add(Conv2D(128, (2, 2), 2, activation="relu"))
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))

else:                   # Without BatchNormalization (Default)
    model_name = 'fashion_mnist.h5'
    history_name = 'historyWithoutBatch'
    model = Sequential()
    model.add(Conv2D(32, (2, 2), activation="sigmoid", input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (2, 2), activation="sigmoid"))
    model.add(Conv2D(128, (2, 2), 2, activation="sigmoid"))
    model.add(Conv2D(32, (2, 2), activation="sigmoid"))
    model.add(Conv2D(64, (2, 2), activation="sigmoid"))
    model.add(Conv2D(128, (2, 2), 2, activation="sigmoid"))
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))

# 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(f_image_train, f_label_train, epochs=10, batch_size=32, validation_data=(f_image_test, f_label_test))
model.summary()
model.save(model_name)
with open(history_name, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
