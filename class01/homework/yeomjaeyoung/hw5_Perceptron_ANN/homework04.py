import numpy as np # forlinear algebra
import matplotlib.pyplot as plt #for plotting things
import os
import scipy
from PIL import Image # for reading images
# Keras Libraries <- CNN
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
#from sklearn.metrics import classification_report, confusion_matrix # <- define evaluation metrics
# MEDICAL IMAGE CLASSIFICATION: BUILD MODEL
# Homework 3 # 
mainDIR = os.listdir('./chest_xray')
print(mainDIR)
train_folder= './chest_xray/train/'
val_folder = './chest_xray/val/'
test_folder = './chest_xray/test/'
# train
os.listdir(train_folder)
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'
#Normal pic
print(len(os.listdir(train_n)))
rand_norm= np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ',norm_pic)
norm_pic_address = train_n+norm_pic
#Pneumonia
rand_p = np.random.randint(0,len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p+sic_pic
print('pneumonia picture title:', sic_pic)
model_fin = Sequential()
#Convolution
model_fin.add(tf.keras.Input(shape = (64, 64, 3)))
model_fin.add(tf.keras.layers.Flatten())
# Fully Connected Layers
model_fin.add(tf.keras.layers.Dense(128, activation='relu'))
# model_fin.add(tf.keras.layers.Dense(64, activation='relu'))
# model_fin.add(tf.keras.layers.Dense(32, activation='relu'))
model_fin.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model_fin.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
num_of_test_samples = 600
batch_size = 32
# Fitting the CNN to the images
# The function ImageDataGenerator augments your image by iterating through image as your CNN is getting ready to process that image
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
#Image normalization.
training_set = train_datagen.flow_from_directory('./chest_xray/train',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
validation_generator = test_datagen.flow_from_directory('./chest_xray/val',target_size=(64, 64),batch_size=32,class_mode='binary')
test_set = test_datagen.flow_from_directory('./chest_xray/test',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
# Homework #4
cnn_model = model_fin.fit(training_set,steps_per_epoch = 163,epochs = 10,validation_data = validation_generator,validation_steps = 624)
test_accu = model_fin.evaluate(test_set,steps=624)
model_fin.save('medical_ann.h5')
# Accuracy
plt.plot(cnn_model.history['accuracy'])
plt.plot(cnn_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validationset'], loc='upper left')
plt.savefig('train_accuracy.png')
plt.show(block=True)
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
plt.show(block=True)
plt.clf()
