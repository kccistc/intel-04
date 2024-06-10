import numpy as np
import cv2

img = cv2.imread('my_input.jpeg')
print(img.shape)

img_expand = np.expand_dims(img, 0)
print(img_expand.shape)

img_T = img_expand.transpose((0,3, 1,2))
print(img_T.shape)