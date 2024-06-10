import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import copy

A = ('A', 'B')

Z = list(A)

Z.append('C')
thistuple = tuple(Z)
print(thistuple)


###############################################################

n = 6

A = np.zeros((n,n))

for j in range(n):
    for i in range(n):
        A[j,i] = j*n+i

print(A) 


#####################################################################


B = A.reshape(-1,)
print(B)


########################################################################


image = Image.open("lena.jpeg")

img_ch_expend = np.expand_dims(image,0) 
print(img_ch_expend.shape)
img_ch_trans = np.transpose(img_ch_expend,(0,3,1,2)) 
print(img_ch_trans.shape)

#########################################################################



# convolution
img = cv2.imread('lena.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
kernel = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])
#print(img.shape[1])
imgcpy = copy.deepcopy(img)
imgcpy2 = np.zeros((imgcpy.shape[0], imgcpy.shape[1], imgcpy.shape[2]), dtype = int) 
#imgcpy2 = np.zeros((imgcpy.shape[0], imgcpy.shape[1]), dtype = int) 

plt.imshow(img/255)



for h in range(2, img.shape[0]-1):
    for w in range(2, img.shape[1]-1):
        #img_tmp = np.dot(imgcpy[h-1:h-1+kernel.shape[0], w-1:w-1+kernel.shape[1]], kernel)
        #imgcpy2[h,w] = np.sum(img_tmp)
        for c in range(1, img.shape[2]):
            img_tmp = np.dot(img[h-1:h-1+kernel.shape[0], w-1:w-1+kernel.shape[1], c], kernel)
            imgcpy2[h,w,c] = np.sum(img_tmp)

#img3 = cv2.cvtColor(imgcpy2, cv2.COLOR_BGR2RGB)
plt.imshow(imgcpy2/255)



# img = cv2.imread('lena.jpeg', cv2.IMREAD_GRAYSCALE)
# print(img)
# kernel = np.array([[1,1,1],[1, -8, 1], [1, 1, 1]])
# print(kernel)
# output = cv2.filter2D(img, -1, kernel)
# cv2.imshow('edge', output)
# cv2.waitKey(0)




