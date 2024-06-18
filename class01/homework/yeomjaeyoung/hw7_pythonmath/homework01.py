import numpy as np

# 실습 1
# 튜플에 요소 추가하는 방법
# 튜플은 고정된 형태이기에 다음과 같은 방법으로만 요소를 추가할 수 있다.
tuple_data = ('A','B')
# 1 . +로 요소를 추가하는 방법
tuple_data = tuple_data+('C',)
print(tuple_data)
#2. list로 요소를 추가하는 방법
tuple_data_list = list(tuple_data)
tuple_data_list.append('D')
tuple_data = tuple(tuple_data_list)
print(tuple_data)


# 실습 2
# 정수 n을 입력받아 nxn크기의 숫자 사각형을 출력
n = input("배열의 숫자를 입력하시오.")
n = int(n)
a = np.zeros((n,n))
num = 1
for j in range(n):
    for i in range(n):
       a[j,i] = j*n+i      
print(a)

# 실습 3
# 실습 2에서 수행한 결과를 reshape을 이용해서 1차원 형태로 변환한다.
b = np.reshape(a,(-1,))
print(b)

# 실습 4
# 임의의 이미지 파일을 불러온다.
import cv2
import numpy as np

img = cv2.imread('image.png')
print(img.shape)
# 축 하나 추가
img_ex = np.expand_dims(img,axis = 0)
print(img_ex.shape)
img_ex = np.transpose(img_ex,(0,3,1,2))
print(img_ex.shape)

import cv2
import numpy as np

# 필터 정의
kernel_edge = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

# 이미지 로드
img = cv2.imread('image.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 이미지의 너비, 높이, 채널 수 가져오기
height, width, channel = img.shape

# 결과 이미지를 저장할 배열 생성
img_out = np.zeros((height, width, channel), dtype=np.uint8)

# 필터 크기
k_h, k_w = kernel_edge.shape
pad_h = k_h // 2
pad_w = k_w // 2

# 이미지에 필터 적용
for c in range(channel):
    for h in range(pad_h, height - pad_h):  # 이미지의 높이
        for w in range(pad_w, width - pad_w):  # 이미지의 너비
            # 필터와 이미지의 일부분을 곱하고 합산
            region = img[h - pad_h:h + pad_h + 1, w - pad_w:w + pad_w + 1, c]
            filtered_value = np.sum(region * kernel_edge)
            # 결과값 클리핑
            filtered_value = np.clip(filtered_value, 0, 255)
            # 결과 이미지에 저장
            img_out[h, w, c] = filtered_value

# 결과 이미지 표시
cv2.imshow('Edge Detection Result', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
