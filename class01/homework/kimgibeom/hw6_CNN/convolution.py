import cv2
import numpy as np

img = cv2.imread('my_input.jpg', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])
kernel_identity = np.array([[0, 0, 0],[0, 1, 0], [0, 0, 0]])
kernel_ridge = np.array([[0, -1, 0],[-1, 4, -1], [0, -1, 0]])
kernel_edge = np.array([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])
kernel_sharpen = np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]])
kernel_boxblur = np.array([[1./9, 1./9, 1./9],[1./9, 1./9, 1./9], [1./9, 1./9, 1./9]])
kernel_Gaussianblur_3 = np.array([[1.16, 1./8, 1./16],[1./8, 1./4, 1./8], [1./16, 1./8, 1./16]])
kernel_Gaussianblur_5 = np.array([[1./256, 4./256, 6./256, 4./256, 1./256],
                                  [4./256, 16./256, 24./256, 16./256, 4./256],
                                  [6./256, 24./256, 36./256, 24./256, 6./256],
                                  [4./256, 16./256, 24./256, 16./256, 4./256],
                                  [1./256, 4./256, 6./256, 4./256, 1./256]])

print(kernel)
output_kernel = cv2.filter2D(img, -1, kernel)
cv2.imshow('edge1', output_kernel)
cv2.waitKey(1)

print(kernel_identity)
output_kernel_identity = cv2.filter2D(img, -1, kernel_identity)
cv2.imshow('edge2', output_kernel_identity)
cv2.waitKey(1)

print(kernel_ridge)
output_kernel_ridge = cv2.filter2D(img, -1, kernel_ridge)
cv2.imshow('edge3', output_kernel_ridge)
cv2.waitKey(1)

print(kernel_edge)
output_kernel_edge = cv2.filter2D(img, -1, kernel_edge)
cv2.imshow('edge4', output_kernel_edge)
cv2.waitKey(1)

print(kernel_sharpen)
output_kernel_sharpen = cv2.filter2D(img, -1, kernel_sharpen)
cv2.imshow('edge5', output_kernel_sharpen)
cv2.waitKey(1)

print(kernel_boxblur)
output_kernel_boxblur = cv2.filter2D(img, -1, kernel_boxblur)
cv2.imshow('edge6', output_kernel_boxblur)
cv2.waitKey(1)

print(kernel_Gaussianblur_3)
output_kernel_Gaussianblur_3 = cv2.filter2D(img, -1, kernel_Gaussianblur_3)
cv2.imshow('edge7', output_kernel_Gaussianblur_3)
cv2.waitKey(1)

print(kernel_Gaussianblur_5)
output_kernel_Gaussianblur_5 = cv2.filter2D(img, -1, kernel_Gaussianblur_5)
cv2.imshow('edge8', output_kernel_Gaussianblur_5)
cv2.waitKey(0)