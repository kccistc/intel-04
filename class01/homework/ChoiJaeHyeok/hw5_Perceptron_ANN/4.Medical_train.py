# Homework 3&4

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
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'
# Normal pic
print(len(os.listdir(train_n)))
rand_norm = np.random.randint(0, len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ', norm_pic)
norm_pic_address = train_n+norm_pic
#Pneumonia
rand_p = np.random.randint(0, len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p+sic_pic
print('pneumonia picture title: ', sic_pic)

# Load the images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

#Let's plt these images
f = plt.figure(figsize = (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Noraml')

a2 = f.add_subplot(1,2,2)
image_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.show()

# Convolution
model_in = Input(shape = (64,64,3))
model = Conv2D(filters = 128, kernel_size = 3, padding = 'valid', activation = 'gelu')(model_in)
model = BatchNormalization()(model)
model = MaxPooling2D(2)(model)
model = Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'gelu')(model)
model = BatchNormalization()(model)
model = MaxPooling2D(2)(model)
model = Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'gelu')(model)
model = BatchNormalization()(model)
model = MaxPooling2D(2)(model)
model = Flatten()(model)
# Fully Connected layers
model = Dense(activation = 'gelu', units = 128)(model)
model = BatchNormalization()(model)
model = Dropout(0.3)(model)
model = Dense(activation = 'gelu', units = 64)(model)
model = BatchNormalization()(model)
model = Dense(activation = 'sigmoid', units = 1)(model)
model_fin = Model(inputs = model_in, outputs = model)

model_fin.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print(model_fin.summary())

# Fitting the CNN to the images
# The function ImageDataBenerator augments your image by iterating through image as your CNN is getting ready to process that image
num_of_test_samples = 600
batch_size= 32
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./chest_xray/train',
                                                 target_size = (64, 64),
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')

validation_set = test_datagen.flow_from_directory('./chest_xray/val/',
                                                  target_size = (64,64),
                                                  batch_size = batch_size,
                                                  class_mode = 'binary')

test_set = test_datagen.flow_from_directory('./chest_xray/test',
                                            target_size = (64, 64),
                                            batch_size = batch_size,
                                            class_mode = 'binary')

history = model_fin.fit(training_set, steps_per_epoch = 100, epochs = 20, validation_data = validation_set, validation_steps = 100)

eval = model_fin.evaluate(test_set, steps = 100)
print("model test accuracy : ", eval[1])

model_fin.save('medical_model.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig('train_accuracy.png')
plt.show(block=False)
plt.clf()

plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'],
loc='upper left')
plt.savefig('train_loss.png')
plt.show(block=False)
plt.clf()