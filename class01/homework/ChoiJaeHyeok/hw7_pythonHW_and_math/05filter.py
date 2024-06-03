import cv2
import numpy as np

def filter_2D(image, kernel):
    img_h,img_w = image.shape[0], image.shape[1]
    kernel_h, kernel_w = kernel.shape[0], kernel.shape[1]
    img_pad = np.zeros((img_h+kernel_h-1, img_w+kernel_w-1))
    img_pad[kernel_h//2:-(kernel_h//2), kernel_w//2:-(kernel_w//2)] =image
    output = np.zeros((img_h, img_w))
    for i in range(img_h):
        for j in range(img_w):
            output[i,j] = np.sum(np.dot(img_pad[i:i+kernel_h,j:j+kernel_w], kernel[:,:]))
    return output


img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])
print(kernel)
output = filter_2D(img, kernel)
cv2.imshow('edge', output)

cv2.waitKey()

