import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./image.png', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])
print(kernel)
output = cv2.filter2D(img,-1,kernel)

k_identity = np.array([[0,0,0],[0,1,0],[0,0,0]])
k_ridge = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
k_edge = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
k_sharp = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
k_box = np.array([[1,1,1],[1,1,1],[1,1,1]])/9.0
k_Gau3 = np.array([[1,2,1],[2,4,2],[1,2,1]])
k_Gau5 = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])

list_kernel = [k_identity, k_ridge, k_edge, k_sharp, k_box, k_Gau3, k_Gau5]
list_name = ['identity', 'ridge', 'edge', 'sharp', 'box', 'Gauss3', 'Gauss5']

output_img = []

for kernel_i in list_kernel:
    new_img = cv2.filter2D(img, -1, kernel_i)
    output_img.append(new_img)

plt.figure(figsize=(15,15))
i=0
for new_output, name in zip(output_img, list_name):
    plt.subplot(2,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(new_output)
    plt.xlabel(name)
    i +=1
plt.show()

cv2.imshow('edge',output)
cv2.waitKey(0)
cv2.destroyAllWindows()


