import cv2
import numpy as  np

def filter2D(img, outSize, kernel):
                 
    if (kernel.shape[0] != 3):
        print("Please check the kernel size. \nIt would be 3x3")
        
        
    if (outSize != -1):
        outImage = np.zeros((img.shape[0]-2, img.shape[1]-2, 3))
        tmp = np.zeros((3, 3))
        
        for rgb in range(3):
            for h in range(img.shape[0]-2):
                for w in range(img.shape[1]-2):
                    tmp = img[h:h+3, w:w+3, rgb] * kernel
                    outImage[h, w, rgb] = np.sum(tmp) / 255
        return outImage
                        
    else:
        outImage = np.zeros((img.shape[0]+2, img.shape[1]+2, 3))
        tmp = np.zeros((3, 3))
        
        for rgb in range(3):
            for h in range(img.shape[0]):
                for w in range(img.shape[1]):
                    outImage[h+1, w+1, rgb] = img[h, w, rgb]
        
        for rgb in range(3):
            for h in range(outImage.shape[0]-4):
                for w in range(outImage.shape[1]-4):
                    tmp = img[h:h+3, w:w+3, rgb] * kernel
                    outImage[h+1, w+1, rgb] = np.sum(tmp) / 255
        return outImage[1:-1, 1:-1, :]


img = cv2.imread('unnamed.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)

kernel = np.array([[-1, -1, -1], [0, 0, 0],  [1, 1, 1]])
print(kernel)

output = filter2D(img, -1, kernel)
cv2.imshow('edge', output)
cv2.waitKey(0)