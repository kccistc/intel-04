import cv2
import numpy as np

img = cv2.imread('my_input.jpeg', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[0, -1, 0], [-1, 4, -1],  [0, -1, 0]])
print(kernel)
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('edge', output)
cv2.waitKey(0)