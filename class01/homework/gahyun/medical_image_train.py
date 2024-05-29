import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

mainDIR = os.listdir('/home/ubuntu/Downloads/archive/chest_xray')
print(mainDIR)
train_folder = '/home/ubuntu/Downloads/archive/chest_xray/train/'
val_folder = '/home/ubuntu/Downloads/archive/chest_xray/val/'
test_folder = '/home/ubuntu/Downloads/archive/chest_xray/test/'

#train
os.listdir(train_folder)
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'

#Normal pic
print(len(os.listdir(train_n)))
rand_norm = np.random.randint(0, len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ', norm_pic)
norm_pic_address = train_n+norm_pic

#Pneumonia
rand_p = np.random.randint(0, len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p + sic_pic
print('pneunonia picture title: ', sic_pic)

#Load Images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

# Plt Images
f = plt.figure(figsize=(10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1,2,2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.show()

# Build CNN model

model=Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size=(2, 2), ))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Dateset Preparation

num_of_test_samples = 600
batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/home/ubuntu/Downloads/archive/chest_xray/train',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory('/home/ubuntu/Downloads/archive/chest_xray/val/'
                                                        , target_size=(64, 64),
                                                          batch_size=32,
                                                          class_mode='binary')
test_set = test_datagen.flow_from_directory('/home/ubuntu/Downloads/archive/chest_xray/test',
                                            target_size = (64, 64), 
                                            batch_size = 32,
                                            class_mode = 'binary')
model.summary()

# Model fit
history = model.fit(training_set, epochs=10,batch_size=10,validation_data=validation_generator)
model.summary()
model.save('medical_ann.h5')

# Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validationset'], loc='upper left')
plt.savefig('train_accuracy.png')
plt.show(block=False)
plt.clf()

# Loss
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.savefig('train_loss.png')
plt.show(block=False)
plt.clf()
