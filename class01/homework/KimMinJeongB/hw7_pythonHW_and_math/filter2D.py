import cv2
import copy
import numpy as np

img = cv2.imread('my_input.jpeg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이 스케일로 변환
kernel = np.array([[1,1,1], [1,-8,1], [1,1,1]])
print(kernel)

imgcpy = copy.deepcopy(gray_img)  # 그레이 스케일 이미지로 복사
imgcpy2 = np.zeros_like(gray_img, dtype=int)  # 결과 이미지를 저장할 배열 초기화

for h in range(1, gray_img.shape[0]-1):
    for w in range(1, gray_img.shape[1]-1):
        img_tmp = np.sum(gray_img[h-1:h-1+kernel.shape[0], w-1:w-1+kernel.shape[1]] * kernel)
        imgcpy2[h,w] = img_tmp

cv2.imshow("edge", imgcpy2/255)
cv2.waitKey(0)