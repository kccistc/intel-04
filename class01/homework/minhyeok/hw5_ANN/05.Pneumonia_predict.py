import numpy as np # forlinear algebra
import matplotlib.pyplot as plt #for plotting things
import os
from PIL import Image # for reading images
# Keras Libraries <- CNN
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

model = tf.keras.models.load_model('./medical_ann.h5')
# 검증 및 테스트 데이터는 정규화만 수행
test_datagen = ImageDataGenerator(rescale = 1./255)
# 테스트용 데이터 셋 로드
test_set = test_datagen.flow_from_directory('./chest_xray/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# 테스트 데이터로 모델 평가
test_steps = test_set.samples // test_set.batch_size
test_accu = model.evaluate(test_set, steps=test_steps)
print('The testing accuracy is :',test_accu[1]*100, '%')

# 예측 수행
# argmax는 가장 큰 값의 인덱스를 반환함
# 전체 테스트 셋에 대해 예측을 수행
# 이진 분류에서 argmax는 불필요
Y_pred = model.predict(test_set, steps=test_steps)
y_pred = (Y_pred > 0.5).astype(int)
class_name = ["NORMAL", "PNEUMONIA"]
print("actual || predict")
for i in range(len(Y_pred)):
    print(class_name[test_set.labels[i]], " || ", class_name[y_pred[i][0]])

# Accuracy
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig('train_accuracy.png')
plt.show(block=False)
plt.clf()

# Loss
plt.plot(model.history['val_loss'])
plt.plot(model.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.savefig('train_loss.png')
plt.show(block=False)
plt.clf()
