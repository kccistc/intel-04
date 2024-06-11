import cv2
import numpy as np

img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
kernel_edge = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])
kernel_identity = np.array([[0, 0, 0],[0, 1, 0], [0, 0, 0]])
kernel_ridge = np.array([[0, -1, 0],[-1, 4, -1], [0, -1, 0]])
kernel_edge_light = np.array([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])
kernel_sharpen = np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]])
kernel_boxblur = np.array([[1./9, 1./9, 1./9],[1./9, 1./9, 1./9], [1./9, 1./9, 1./9]])
kernel_Gaussianblur_3 = np.array([[1./16, 2./16, 1./16],[2./16, 4./16, 2./16], [1./16, 2./16, 1./16]])
kernel_Gaussianblur_5 = np.array([[1./256, 4./256, 6./256, 4./256, 1./256],
                                  [4./256, 16./256, 24./256, 16./256, 4./256],
                                  [6./256, 24./256, 36./256, 24./256, 6./256],
                                  [4./256, 16./256, 24./256, 16./256, 4./256],
                                  [1./256, 4./256, 6./256, 4./256, 1./256]])
kernel_unsharp_5 = -np.array([[1./256, 4./256, 6./256, 4./256, 1./256],
                                  [4./256, 16./256, 24./256, 16./256, 4./256],
                                  [6./256, 24./256, -476./256, 24./256, 6./256],
                                  [4./256, 16./256, 24./256, 16./256, 4./256],
                                  [1./256, 4./256, 6./256, 4./256, 1./256]])


print(kernel_edge)
output_edge = cv2.filter2D(img, -1, kernel_edge)
cv2.imshow('output_edge', output_edge)
cv2.waitKey(1)
output_identity = cv2.filter2D(img, -1, kernel_identity)
cv2.imshow('output_identity', output_identity)
cv2.waitKey(1)
output_ridge = cv2.filter2D(img, -1, kernel_ridge)
cv2.imshow('output_ridge', output_ridge)
cv2.waitKey(1)
output_edge_light = cv2.filter2D(img, -1, kernel_edge_light)
cv2.imshow('output_edge_light', output_edge_light)
cv2.waitKey(1)
output_sharpen = cv2.filter2D(img, -1, kernel_sharpen)
cv2.imshow('output_sharpen', output_sharpen)
cv2.waitKey(1)
output_boxblur = cv2.filter2D(img, -1, kernel_boxblur)
cv2.imshow('output_boxblur', output_boxblur)
cv2.waitKey(1)
output_Gaussianblur_3 = cv2.filter2D(img, -1, kernel_Gaussianblur_3)
cv2.imshow('output_Gaussianblur_3', output_Gaussianblur_3)
cv2.waitKey(1)
output_Gaussianblur_5 = cv2.filter2D(img, -1, kernel_Gaussianblur_5)
cv2.imshow('output_Gaussianblur_5', output_Gaussianblur_5)
cv2.waitKey(1)
output_unsharp_5 = cv2.filter2D(img, -1, kernel_unsharp_5)
cv2.imshow('output_unsharp_5', output_unsharp_5)
cv2.waitKey(0)