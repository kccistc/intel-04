import cv2
import numpy as np

img = cv2.imread('cat.jpg')
print(img.shape)

exp_img = np.expand_dims(img,axis=0)
print(exp_img.shape)

tran_img = exp_img.transpose((0, 3, 1, 2))
print(tran_img.shape)