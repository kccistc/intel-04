import numpy as np

A = ('A','B')
C = A + ('C',)
print(C)

D = list(A)
D.append('C')
D = tuple(D)
print(D)
