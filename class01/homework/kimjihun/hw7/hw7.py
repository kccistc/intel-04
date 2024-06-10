import numpy as np 
import cv2
# 실습1
A = ('A','B')

A = A +('C',)

print(A)

#실습2
n = 6
A = np.zeros((n,n))

for j in range(n):
    for i in range(n):
        A[j,i]=j*n+i
print(A)
#실습3
B = np.reshape(A,n*n)

print(B)
#실습4
img = cv2.imread('my_input.jpeg')

print(img.shape)

img_ex = np.expand_dims(img,0)

print(img_ex.shape)

img_ex_T = np.transpose(img_ex , (0,3,1,2))

print(img_ex_T.shape)

#실습5
img = cv2.imread('my_input.jpeg',cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])

print(kernel)

output=np.zeros_like(img)

width, height = img.shape

img_pad = np.pad(img,((1,1),(1,1)),mode='constant')

for i in range(width):
    for k in range(height):
        output[i-1][k-1]=np.sum(kernel*img_pad[i-1:i+2,j-1:j+2])


cv2.imshow('edge',output)
cv2.waitKey(0)
