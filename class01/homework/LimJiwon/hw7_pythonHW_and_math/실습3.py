import numpy as np

n = int(input())

A = np.zeros((n,n), dtype = int)

for j in range(n):
    for i in range(n):
        A[j,i] = n * j + i + 1

print(A.reshape(-1,))