import numpy as np # 선형 대수학을 위한 라이브러리
import matplotlib.pyplot as plt # 그래프를 그리기 위한 라이브러리
import os
from PIL import Image # 이미지를 읽기 위한 라이브러리
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

# 디렉토리 경로 정의
mainDIR = os.listdir('./chest_xray')
print(mainDIR)
train_folder= './chest_xray/train/'
val_folder = './chest_xray/val/'
test_folder = './chest_xray/test/'

os.listdir(train_folder)
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'

# 랜덤한 정상 이미지 출력
print(len(os.listdir(train_n)))
rand_norm = np.random.randint(0, len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('Normal picture title:', norm_pic)
norm_pic_address = train_n + norm_pic

# 랜덤한 폐렴 이미지 출력
rand_p = np.random.randint(0, len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_p]
sic_address = train_p + sic_pic
print('Pneumonia picture title:', sic_pic)

# 이미지를 불러오기
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

# 이미지를 시각화
f = plt.figure(figsize=(10, 6))
a1 = f.add_subplot(1, 2, 1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')
a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.show()

# 이미지 증강을 위한 ImageDataGenerator 정의
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(train_folder,
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

# CNN 모델 정의
model_fin = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model_fin.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# 모델 요약 출력
model_fin.summary()

# 모델 훈련
history = model_fin.fit(training_set,
                        epochs=10,
                        batch_size = 32)


# 모델 저장
model_fin.save('chest_xray_cnn_model.h5')
