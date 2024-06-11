import cv2
import numpy as np

img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE) # 640, 612
kernel = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])
print(kernel)

h = img.shape[0]
w = img.shape[1]

output = np.zeros((h, w))

for i in range(1, h - 1):
    for j in range(1, w - 1):
        tmp = img[i-1:i+2, j-1:j+2]

        output[i-1, j-1] = np.sum(tmp * kernel)

cv2.imshow('edge', output)
cv2.waitKey(0)