import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

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

# Homework 4
num_of_test_samples = 600
batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('./chest_xray/train', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory('./chest_xray/val/', target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('./chest_xray/test', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
model_fin.summary()

# tensorflow 버전 업그레이드 -> teps_per_epoch, validation_steps 적용 시, accuracy: 0.0000e+00 오류
# cnn_model = model_fin.fit(training_set, steps_per_epoch = 163, epochs = 10, validation_data = validation_generator, validation_steps = 624)
cnn_model = model_fin.fit(training_set, epochs = 10, validation_data = validation_generator)

test_accu = model_fin.evaluate(test_set,steps=624)

# Homework 6
# Accuracy
plt.plot(cnn_model.history['accuracy'])
plt.plot(cnn_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig('train_accuracy.png')
plt.show(block=False)
plt.clf()

# Loss
plt.plot(cnn_model.history['val_loss'])
plt.plot(cnn_model.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'],
loc='upper left')
plt.savefig('train_loss.png')
plt.show(block=False)
plt.clf()