import os
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, \
Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 데이터셋 경로
mainDIR = os.listdir('./chest_xray')
print(mainDIR)
train_folder = './chest_xray/train/'
val_folder = './chest_xray/val/'
test_folder = './chest_xray/test/'

os.listdir(train_folder)
train_n = train_folder + 'NORMAL/'
train_p = train_folder + 'PNEUMONIA/'

# 정상 이미지 파일 확인
print(len(os.listdir(train_n)))
rand_norm = np.random.randint(0, len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ', norm_pic)
norm_pic_address = train_n + norm_pic

# 폐렴 이미지 파일 확인
rand_p = np.random.randint(0, len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p + sic_pic
print('pneumonia picture title:', sic_pic)

# 이미지 표시
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

f = plt.figure(figsize=(10, 6))
a1 = f.add_subplot(1, 2, 1)
img_plot = plt.imshow(norm_load, cmap='gray')
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load, cmap='gray')
a2.set_title('Pneumonia')
plt.show()

# 모델 생성
model_fin = Sequential()
model_fin.add(Flatten())
model_fin.add(Dense(128, activation='relu', input_dim=64*64))
model_fin.add(Dense(64, activation='relu'))
model_fin.add(Dense(1, activation='sigmoid'))  # 이진 분류

model_fin.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

num_of_test_samples = 600
batch_size = 32

# 데이터 셋 생성
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('./chest_xray/train', 
                                                 target_size=(64, 64), 
                                                 batch_size=32, 
                                                 class_mode='binary')
validation_generator = test_datagen.flow_from_directory('./chest_xray/val/', 
                                                        target_size=(64,64), 
                                                        batch_size=32, 
                                                        class_mode='binary')
test_set = test_datagen.flow_from_directory('./chest_xray/test', 
                                            target_size=(64, 64), 
                                            batch_size=32, 
                                            class_mode='binary')

# 모델 학습
history = model_fin.fit(training_set, epochs=10, batch_size=batch_size, 
          validation_data=validation_generator)
model_fin.summary()
model_fin.save('pneumonia.h5')

plt.clf()
plt.figure(figsize=(8,3))

# 훈련 손실 그래프
plt.subplot(121)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'val'], loc='upper right')

# 훈련 정확도 그래프
plt.subplot(122)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'val'], loc='lower right')
plt.tight_layout()
plt.show()
