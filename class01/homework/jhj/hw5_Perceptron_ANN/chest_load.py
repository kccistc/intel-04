import os
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

# 예측할 이미지가 있는 폴더 경로
test_normal_dir = './chest_xray/test/NORMAL/'
test_pneumonia_dir = './chest_xray/test/PNEUMONIA/'

model = tf.keras.models.load_model('./chest_xray_cnn_model.h5')

# NORMAL 이미지 예측
image_paths = []
for filename in os.listdir(test_normal_dir):
    image_paths.append((os.path.join(test_normal_dir, filename), "정상"))
for filename in os.listdir(test_pneumonia_dir):
    image_paths.append((os.path.join(test_pneumonia_dir, filename), "폐렴"))

# 이미지 경로 리스트를 랜덤하게 섞기
random.shuffle(image_paths)

# 상위 100개의 이미지만 테스트
image_paths = image_paths[:100]

correct_predictions = 0

# 이미지 예측과 결과 출력
print("실제, [예측결과]")
for image_path, label in image_paths:
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batch)
    if prediction[0][0] < 0.5:
        predicted_label = "정상"
    else:
        predicted_label = "폐렴"
    print(f"{label}, [{predicted_label}]")
    
    # 정확도 계산
    if label == predicted_label:
        correct_predictions += 1

accuracy = correct_predictions / len(image_paths)
print(f"\n정확도: {accuracy * 100:.2f}%")