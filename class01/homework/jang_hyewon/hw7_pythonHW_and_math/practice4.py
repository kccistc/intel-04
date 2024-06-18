import numpy as  np
import cv2 

image = cv2.imread("unnamed.jpg")
print(image.shape)

re_image = np.expand_dims(image, 0)
print(re_image.shape)

re2_image = re_image.transpose((0, 3, 1, 2))
print(re2_image.shape)