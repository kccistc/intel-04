import cv2
import numpy as np

img = cv2.imread('image.jpg')
print(img.shape)
img_bhwc = img.reshape(1,img.shape[0],img.shape[1],-1)
img_expend = np.expand_dims(img, 0)
print(img_bhwc.shape)
print(img_expend.shape)
img_bcwh = img_bhwc.reshape(1,-1,img_bhwc.shape[1],img_bhwc.shape[2])
img_expend_T = img_expend.transpose((0,3,1,2))
print(img_bcwh.shape)
print(img_expend_T.shape)


