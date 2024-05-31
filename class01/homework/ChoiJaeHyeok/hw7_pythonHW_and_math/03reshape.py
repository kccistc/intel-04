import numpy as np

n = 6
arr = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        arr[i,j]= n*i+j
print(arr)
arr = arr.reshape(-1)
print(arr)
