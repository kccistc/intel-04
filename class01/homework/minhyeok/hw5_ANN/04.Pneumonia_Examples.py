import numpy as np # forlinear algebra
import matplotlib.pyplot as plt #for plotting things
import os
from PIL import Image # for reading images
# Keras Libraries <- CNN
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

'''
데이터를 랜덤으로 섞고, 확인하는 작업
'''

# 데이터 찾기
mainDIR = os.listdir('./chest_xray')
print(mainDIR)
train_folder= './chest_xray/train/'
val_folder = './chest_xray/val/'
test_folder = './chest_xray/test/'

# 학습 준비
os.listdir(train_folder)
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'

# Normal 이미지 랜덤 쌓기
print(len(os.listdir(train_n)))
rand_norm= np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ',norm_pic)
# 경로 추출
norm_pic_address = train_n+norm_pic

# Pneumonia 이미지 랜덤 쌓기(동일)
rand_p = np.random.randint(0,len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_p]
sic_address = train_p+sic_pic
print('pneumonia picture title:', sic_pic)

# 이미지 로드
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

# 이미지들 확인
f = plt.figure(figsize= (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')

plt.show()

# CNN 모델 구축

#cnn = Sequential()
'''
Sequential()을 굳이 해줄 필요는 없다.
Sequential 모델은 간단하게 층을 쌓아 올리는 방식으로 모델을 만들 때 사용됨.
반면에 Model API는 더 유연한 모델을 정의할 수 있게 해줌.
두 모델 구축 방식에는 장단점이 있는데, 이번에는 Model API를 활용!
'''
# 입력 레이어 정의
model_in = Input(shape = (64, 64, 3))

# Convolutional 레이어 추가
model = Conv2D(32, (3, 3), activation='relu')(model_in)
model = MaxPooling2D((2, 2))(model)

model = Conv2D(64, (3, 3), activation='relu')(model)
model = MaxPooling2D((2, 2))(model)

model = Conv2D(128, (3, 3), activation='relu')(model)
model = MaxPooling2D((2, 2))(model)

# 평탄화 작업
model = Flatten()(model)

# Dense 레이어 추가
model = Dense(128, activation='relu')(model)

# Dropout 레이어 추가
model = Dropout(0.5)(model)

# Output 레이어 추가
model_out = Dense(1, activation='sigmoid')(model)

# 모델 정의(입출력 저장)
model_fin = Model(inputs=model_in, outputs=model_out)

# 컴파일!
model_fin.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# 이미지 데이터 생성기 정의 (학습용 데이터 증강)
# + 데이터를 더욱 다양하게!
# + 오버피팅 방지!
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# 검증 및 테스트 데이터는 정규화만 수행
test_datagen = ImageDataGenerator(rescale = 1./255) #Image normalization.

# 학습용 데이터 셋 로드
# flow_from_directory 이거는 ImageDataGenerator 이 클래스의 메서드임
# 디렉토리 구조를 기반으로 이미지 로드해서, 전처리 및 배치 생성!
training_set = train_datagen.flow_from_directory('./chest_xray/train', 
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# 검증용 데이터 셋 로드
validation_generator = test_datagen.flow_from_directory('./chest_xray/val/',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

# 테스트용 데이터 셋 로드
test_set = test_datagen.flow_from_directory('./chest_xray/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# 모델 구조 출력
print(model_fin.summary())

# steps_per_epoch 및 validation_steps 계산
steps_per_epoch = training_set.samples // training_set.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

# 모델 학습
cnn_model = model_fin.fit(training_set,
                          steps_per_epoch = steps_per_epoch,
                          epochs = 10,
                          validation_data = validation_generator,
                          validation_steps = validation_steps)

# 테스트 데이터로 모델 평가
test_steps = test_set.samples // test_set.batch_size
test_accu = model_fin.evaluate(test_set, steps=test_steps)

# 모델 저장
model_fin.save('medical_ann.h5')
print('The testing accuracy is :',test_accu[1]*100, '%')

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
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.savefig('train_loss.png')
plt.show(block=False)
plt.clf()