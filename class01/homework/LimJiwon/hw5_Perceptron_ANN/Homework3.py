import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

mainDIR = os.listdir('./chest_xray')
print(mainDIR)

train_folder = './chest_xray/train/'
val_folder = './chest_xray/val/'
test_folder = './chest_xray/test/'

# train
os.listdir(train_folder)
train_n = train_folder + 'NORMAL/'
train_p = train_folder + 'PNEUMONIA/'

# Normal pic
print(len(os.listdir(train_n)))
rand_norm = np.random.randint(0, len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ', norm_pic)
norm_pic_address = train_n + norm_pic

#P neumonia
rand_p = np.random.randint(0, len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p + sic_pic
print('pneumonia picture title:', sic_pic)

norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

f = plt.figure(figsize= (10, 6))
a1 = f.add_subplot(1, 2, 1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.show()

# Homework 3
model_fin = Sequential()

model_fin.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model_fin.add(MaxPooling2D(pool_size = (2, 2)))

model_fin.add(Conv2D(32, (3, 3), activation='relu'))
model_fin.add(MaxPooling2D(pool_size = (2, 2)))

model_fin.add(Flatten())

model_fin.add(Dense(128, activation='relu'))
model_fin.add(Dense(1, activation='sigmoid'))

model_fin.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])