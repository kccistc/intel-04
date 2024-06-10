import cv2
import numpy as np

img = cv2.imread('lena.jpeg', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1,1,1],[1, -8, 1], [1, 1, 1]])
print(kernel)
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('edge', output)
cv2.waitKey(0)

