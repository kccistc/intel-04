import cv2
import numpy as np

img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
print(kernel)

output = cv2.filter2D(img, -1, kernel)
cv2.imshow('edge', output)
cv2.waitKey(0)

# Homework 1
identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
ridge = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
edge = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
box = np.array([[1./9, 1./9, 1./9], [1./9, 1./9, 1./9], [1./9, 1./9, 1./9]])
gaus_3 = 1./16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
gaus_5 = 1./256 * np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])

# Identity
print(identity)
output_identity = cv2.filter2D(img, -1, identity)
cv2.imshow('Identity', output_identity)
cv2.waitKey(0)

# Ridge detection
print(ridge)
output_ridge = cv2.filter2D(img, -1, ridge)
cv2.imshow('Ridge detection', output_ridge)
cv2.waitKey(0)

# Edge detection
print(edge)
output_edge = cv2.filter2D(img, -1, edge)
cv2.imshow('Edge detection', output_edge)
cv2.waitKey(0)

# Sharpen
print(sharp)
output_sharp = cv2.filter2D(img, -1, sharp)
cv2.imshow('Sharpen', output_sharp)
cv2.waitKey(0)

# Box blur
print(box)
output_box = cv2.filter2D(img, -1, box)
cv2.imshow('Box blur', output_box)
cv2.waitKey(0)

# Gaussian blur 3x3
print(gaus_3)
output_gaus_3 = cv2.filter2D(img, -1, gaus_3)
cv2.imshow('Gaussian blur 3x3', output_gaus_3)
cv2.waitKey(0)

# Gaussian blur 5x5
print(gaus_5)
output_gaus_5 = cv2.filter2D(img, -1, gaus_5)
cv2.imshow('Gaussian blur 5x5', output_gaus_5)
cv2.waitKey(0)