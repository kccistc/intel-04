import os
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img


# 테스트 데이터 셋 생성
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('./chest_xray/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# 모델 불러오기
model_fin = keras.models.load_model('./pneumonia.h5')
test_accu = model_fin.evaluate(test_set, steps=624)
print('The testing accuracy is :',test_accu[1]*100, '%')

# label 얻기 
labels = test_set.labels

# 테스트 데이터 예측
Y_pred = model_fin.predict(test_set)
#y_pred = np.argmax(Y_pred)
y_pred = []
for yy in Y_pred:
    if yy >= 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

# 결과 출력
class_name = ["NORMAL", "PNEUMONIA"]
print("%-10s" % "actual", "|| ", "%-10s" % "predict")
for i in range(624):
    if i%10 == 0:
        print("%-10s" % class_name[labels[i]], end=" || ")
        print("%-10s" % class_name[y_pred[i]], end='\n')