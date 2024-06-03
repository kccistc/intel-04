import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt


'''
튜플형 데이터 (A,B)를 만들고 C를 추가하기.
''' 
A = ('A', 'B')
print(A)
C = A + ('C',)
print(C)



'''
정수 n을 입력받아 n x n 크기의 숫자 사각형을 출력.
''' 
print()
# 나의 답안 #
# n = 6 # 정수 n을 입력 받았다고 가정
# num = 0
# for i in range(n):
#     for k in range(n):
#         num += 1
#         print(num, end=' ')
#     print()

# 교수님 답안
n = 6
A = np.zeros((n,n))
for j in range(n):
    for i in range(n):
        A[j,i] = j*n+i
print(A)


'''
실습 1(위)에서 수행한 결과를 reshape을 이용해서 1차원 형태로 변환한다.
''' 
print()
# 배열로 저장한 교수님 답안 채택

A1 = A.reshape(-1)
A2 = A.flatten()
print(A1)
print(A2)


'''
1. 임의의 이미지 파일을 불러온다.
2. Numpy의 expend_dims를 사용해서 이미지의 파일의 차원을 하나 더 늘려(Height, Width, Channel)을 (Batch, Height, Width, Channel)로 확장한다. (이미지 출력 불필요)
3. Numpy의 transpose를 이용해서 차원의 순서를 (Batch, Width, Height, Channel)에서 (Bath, channel, width, height)로 변경한다.(이미지 출려 불필요)
해당 결과는 imge.shape를 통해 결과를 확인한다.
'''
print()
# 1. 이미지 파일 불러오기
img = cv2.imread('hw6_CNN/dog.jpeg')
print("처음 이미지 shape:", img.shape)

# 2. 차원 확장 (Batch, Height, Width, Channel)
expanded_img = np.expand_dims(img, axis=0)
# 교수님 답안
# expanded_img = np.expand_dims(img, 0)
print("차원 확장 이미지 shape:", expanded_img.shape)

# 3. 차원 순서 변경 (Batch, Channel, Width, Height)
transposed_img = np.transpose(expanded_img, (0, 3, 1, 2))
# 교수님 답안
# img_expend_T = expanded_img.transpose((0, 3, 1, 2))

print("순서 변경 이미지 shape:", transposed_img.shape)


'''
img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])
print(kernel)
### output = cv2.filter2D(img, -1, kernel) ### 
### ### 코드 부분을 직접 구현하기
'''
img = cv2.imread('hw6_CNN/dog.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
kernel = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])
imgcpy = copy.deepcopy(img)
imgcpy2 = np.zeros((imgcpy.shape[0], imgcpy.shape[1], imgcpy.shape[2]))

# for h in range(2, img.shape[0]-1):
#     for w in range(2, img.shape[1]-1):
#         for c in range(1, img.shape[2]):
#             img_tmp = np.dot(img[h-1:h-1+kernel.shape[0], w-1:w-1+kernel.shape[1], c], kernel)
#             imgcpy2[h,w,c] = np.sum(img_tmp)

# plt.imshow(imgcpy2/255)

# 커널을 적용한 결과를 저장
for h in range(1, img.shape[0] - 1):
    for w in range(1, img.shape[1] - 1):
        for c in range(img.shape[2]):
            imgcpy2[h, w, c] = np.sum(img[h-1:h+2, w-1:w+2, c] * kernel)

# 결과 클리핑 [0, 255] 범위로
'''
클리핑 하는 이유는, 현재 클리핑 오류가 발생하여
값이 주어진 범위를 벗어나는 경우라고 생각함.
그래서 범위 유지를 위해서 클리핑 적용
'''
imgcpy2 = np.clip(imgcpy2, 0, 255)

# 데이터 타입 변환 (np.uint8)
imgcpy2 = imgcpy2.astype(np.uint8)

# 결과 출력
plt.imshow(imgcpy2)
plt.show()