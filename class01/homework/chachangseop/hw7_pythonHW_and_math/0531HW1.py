import cv2
import numpy as np
#튜플형 데이터 (A,B)를 만들고 C를 추가한다
A=('A','B')
C = A + ('C',)
print(C)
print(type(C))

#실습2 정수 n 을 입력받아 n x n 크기의 숫자 사각형을 출력
import numpy as np

# 1234
# 5678
# 9 10 11 12
# 13 14 15 16

n=6
A =  np.zeros((n,n))
for j in range(n):
    for i in range(n):
            A[j,i]=j*n+i
            
print(A)
 


# n=4
# def print_number_square(n):
#     number = 1
#     for i in range(n):
#         for j in range(n):
#             print(f"{number:2}", end=" ")
#             number += 1
#         print()
# print_number_square(n)
#실습3 
#실습1에서 수행한 결과를 reshape을 이용해서 1차원 형태로 변환
print(A.reshape(-1,))
#실습4
#1. 임의의 이미지 파일을 불러온다.
# 2. Numpy의 expend_dims를 사용해서 이미지 파일의 차원을 하나
# 더 늘려 (Height, Width, Channel)을 (Batch, Height, Width,
# Channel)로 확장한다. (이미지 출력 불필요)
# 3. Numpy의 transpose를 이용해서 차원의 순서를 (Batch, Width,
# Height, Channel)에서 (Batch, channel, width, height) 로 변경한다.
# (이미지 출력 불필요)
# 해당 결과는 imge.shape를 통해 결과를 확인한다.

import cv2
img = cv2.imread('./lena5.jpg')
print(img.shape)
img_expend = np.expand_dims(img, axis=0)
print(img_expend.shape)
img_expend_T=img_expend.transpose((0,3,1,2))
print(img_expend_T.shape)



# import cv2
# import numpy as np
# img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
# kernel = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])
# print(kernel)
# output = cv2.filter2D(img, -1, kernel)
# cv2.imshow('edge', output)
# cv2.waitKey(0)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
img = cv2.imread('./lena5.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
kernel = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])
imgcpy = copy.deepcopy(img)
imgcpy2 = np.zeros((imgcpy.shape[0],imgcpy.shape[1],imgcpy.shape[2]))

for h in range(2, img.shape[0]-1):
    for w in range(2, img.shape[1]-1):
        for c in range(1, img.shape[2]):
            img_tmp = np.dot(
                img[h-1:h-1+kernel.shape[0],w-1:w-1+kernel.shape[1],c],kernel
            )
            imgcpy2[h,w,c]=np.sum(img_tmp)


cv2.imshow('edge', imgcpy2/255)
cv2.waitKey(1)