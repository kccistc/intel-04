import numpy as np # forlinear algebra
import matplotlib.pyplot as plt #for plotting things
import os
from PIL import Image # for reading images
# Keras Libraries <- CNN
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
#from sklearn.metrics import classification_report, confusion_matrix # <- define evaluation metrics
# MEDICAL IMAGE CLASSIFICATION: BUILD MODEL
# Homework 5 # 
# 이진 변환 함수 정의 = > 시그모이드의 0,1 출력을 위함
def binary_threshold(predictions, threshold=0.5):
    return (predictions >= threshold).astype(int)
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
#Image normalization.
training_set = train_datagen.flow_from_directory('./chest_xray/train/',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
validation_generator = test_datagen.flow_from_directory('./chest_xray/val/',target_size=(64, 64),batch_size=32,class_mode='binary')
test_set = test_datagen.flow_from_directory('./chest_xray/test/',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
cnn_model = tf.keras.models.load_model('./medical_ann.h5')
num = 30
test_images, test_labels = next(test_set)  # 테스트 세트에서 배치 가져오기
predict = cnn_model.predict(test_images[:num])
class_names = ["Normal","Pneumonia"]
predict_binary = 0
for i in range(num):
    predict_binary = binary_threshold(predict[i][0])
    test_binary = binary_threshold(test_labels[i])
    # 출력 줄맞춤을 위한 폭 설정
    max_length = 10
    # 왼쪽 정렬하여 출력
    print(f"Predict : {class_names[(predict_binary)]:<{max_length}}  ||  Actual : {class_names[(test_binary)]:<{max_length}}")