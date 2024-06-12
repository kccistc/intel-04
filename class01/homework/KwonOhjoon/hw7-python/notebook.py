import cv2
import numpy as np


# 문제 1
print('문제 1')
tuple_1 = ('A', 'B')
tuple_2 = tuple_1 + ('C',)
print(tuple_2)
print('\n')

# 문제 2
print('문제 2')
n = 4
a = []
for i in range(n):
    a.append([(n*i) + (j+1) for j in range(n)])
a = np.array(a)
print(a)
print('\n')

# 문제 3
print('문제 3')
a = np.array(a)
print(a.reshape(-1,))
print('\n')

# 문제 4
print('문제 4')
img = cv2.imread('convolution/dogbird.png')
print(img.shape)

expaned_img = np.expand_dims(img, 0)
print(expaned_img.shape)

transposed_img = expaned_img.transpose((0, 3, 2, 1))
print(transposed_img.shape)
print('\n')

# 문제 5
print('문제 5')
img = cv2.imread('convolution/dogbird.png', cv2.IMREAD_GRAYSCALE)
mask = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
mask_size = 3

img_h, img_w = img.shape
mask_h, mask_w = mask.shape

pad_size = mask_size // 2
pad_img = np.zeros((img_h + pad_size*2, img_w + pad_size*2))
pad_img[pad_size:pad_size+img_h, pad_size:pad_size+img_w] = img

result_img = np.zeros((img_h, img_w))

for h in range(img_h):
    for w in range(img_w):
        result_img[h, w] = np.sum(pad_img[h:h+mask_h, w:w+mask_w] * mask)

result_img = np.clip(result_img, 0, 255)
result_img = result_img.astype(np.uint8)
cv2.imshow('result', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
