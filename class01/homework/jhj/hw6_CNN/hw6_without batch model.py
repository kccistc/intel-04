import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
import pickle

# 패션 MNIST 데이터셋 로드
fashion_mnist = tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()

# 이미지 정규화
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

# 클래스 이름 정의
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 이미지와 라벨 시각화
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[f_label_train[i]])
plt.show()

# 이미지 데이터 형태 변환 (CNN 입력을 위해 4차원 배열로 변환)
f_image_train = f_image_train.reshape(-1, 28, 28, 1)
f_image_test = f_image_test.reshape(-1, 28, 28, 1)

# 라벨을 one-hot 인코딩
f_label_train = tf.keras.utils.to_categorical(f_label_train, 10)
f_label_test = tf.keras.utils.to_categorical(f_label_test, 10)

# CNN 모델 정의
model = Sequential()
model.add(Conv2D(32, (2, 2), activation="sigmoid", input_shape=(28, 28, 1)))
model.add(Conv2D(64, (2, 2), activation="sigmoid"))
model.add(Conv2D(128, (2, 2), activation="sigmoid", strides=2))
model.add(Conv2D(32, (2, 2), activation="sigmoid"))
model.add(Conv2D(64, (2, 2), activation="sigmoid"))
model.add(Conv2D(128, (2, 2), activation="sigmoid", strides=2))

# ANN 레이어 추가
model.add(Flatten())
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

# 모델 컴파일
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# 모델 훈련
history = model.fit(f_image_train, f_label_train, validation_data=(f_image_test, f_label_test), epochs=30, batch_size=256)

# 모델 요약
model.summary()

# 모델 저장
model.save('fashion_mnist.h5')

# 학습 기록 저장
with open('historyBatchReLu', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
