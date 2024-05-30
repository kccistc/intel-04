# 기본
import cv2
import numpy as np
img = cv2.imread('cat01_256.png', cv2.IMREAD_GRAYSCALE)
kernel1 = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])
print(kernel1)
output1 = cv2.filter2D(img, -1, kernel1)

# identity
kernel2 = np.array([[0, 0, 0],[0, 1, 0], [0, 0, 0]])
# print(kernel2)
output2 = cv2.filter2D(img, -1, kernel2)


# Edge detection 1
kernel3 = np.array([[0, -1, 0],[-1, 4, -1], [0, -1, 0]])
# print(kernel2)
output3 = cv2.filter2D(img, -1, kernel3)

# Edge detection 2
kernel4 = np.array([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])
# print(kernel2)
output4 = cv2.filter2D(img, -1, kernel4)

# Sharpen
kernel5 = np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]])
# print(kernel2)
output5 = cv2.filter2D(img, -1, kernel5)

# Box blur
kernel6 = np.array([[1/9., 1/9., 1/9.],[1/9., 1/9., 1/9.], [1/9., 1/9., 1/9.]])
# print(kernel2)
output6 = cv2.filter2D(img, -1, kernel6)

# Gaussian blur
kernel7 = np.array([[1/16., 2/16., 1/16.],[2/16., 4/16., 2/16.], [1/16., 2/16., 1/16.]])
# print(kernel2)
output7 = cv2.filter2D(img, -1, kernel7)


# Gaussian blur2
kernel8 = np.array([[1/256., 4/256., 6/256., 4/256., 1/256.],
		[4/256., 16/256., 24/256., 16/256., 4/256.],
		[6/256., 24/256., 36/256., 24/256., 6/256.],
		[4/256., 16/256., 24/256., 16/256., 4/256.],
		[1/256., 4/256., 6/256., 4/256., 1/256.]])
# print(kernel2)
output8 = cv2.filter2D(img, -1, kernel8)

cv2.imshow('edge', output1)
cv2.imshow('Identity', output2)
cv2.imshow('Edge1', output3)
cv2.imshow('Edge2', output4)
cv2.imshow('Sharpen', output5)
cv2.imshow('Blur', output6)
cv2.imshow('Gaussian', output7)
cv2.imshow('Gaussian2', output8)

cv2.waitKey(0)
cv2.destroyAllWindows()

