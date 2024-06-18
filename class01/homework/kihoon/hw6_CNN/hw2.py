import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense
import pickle
from tensorflow.keras.utils import to_categorical

# Fashion MNIST 데이터셋 불러오기
fashion_mnist = tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

# 라벨을 원-핫 인코딩
f_label_train = to_categorical(f_label_train)
f_label_test = to_categorical(f_label_test)

# 모델 컴파일 및 학습, 모델 저장 및 학습 이력 저장하는 함수 정의
def compile_and_save(model, count): 
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])
    # 모델 학습
    history = model.fit(f_image_train, f_label_train, epochs=10, batch_size=100, validation_data=(f_image_test, f_label_test))
    model.summary()
    # 모델 저장
    model.save('fashion_mnist_%d.h5' % count)
    # 학습 이력 저장
    with open('historyBatchReLu%d.pickle' % count, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

# Without Batch
model = Sequential([
    Conv2D(32, (2, 2), activation="sigmoid", input_shape=(28, 28, 1)),
    Conv2D(64, (2, 2), activation="sigmoid"),
    Conv2D(128, (2, 2), activation="sigmoid"),
    Conv2D(32, (2, 2), activation="sigmoid"),
    Conv2D(64, (2, 2), activation="sigmoid"),
    Conv2D(128, (2, 2), activation="sigmoid"),
    Flatten(),
    Dense(128, activation="sigmoid"),
    Dense(10, activation="softmax")
])

compile_and_save(model, 1)

# With Batch
model = Sequential([
    Conv2D(32, (2, 2), activation="sigmoid", input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(64, (2, 2), activation="sigmoid"),
    BatchNormalization(),
    Conv2D(128, (2, 2), strides=2, activation="sigmoid"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(32, (2, 2), activation="sigmoid"),
    BatchNormalization(),
    Conv2D(64, (2, 2), activation="sigmoid"),
    BatchNormalization(),
    Conv2D(128, (2, 2), strides=2, activation="sigmoid"),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation="sigmoid"),
    Dense(10, activation="softmax")
])

compile_and_save(model, 2)

# With ReLU
model = Sequential([
    Conv2D(32, (2, 2), activation="relu", input_shape=(28, 28, 1)),
    Conv2D(64, (2, 2), activation="relu"),
    Conv2D(128, (2, 2), strides=2, activation="relu"),
    Conv2D(32, (2, 2), activation="relu"),
    Conv2D(64, (2, 2), activation="relu"),
    Conv2D(128, (2, 2), strides=2, activation="relu"),
    Flatten(),
    Dense(128, activation="sigmoid"),
    Dense(10, activation="softmax")
])

compile_and_save(model, 3)