import numpy as np


def print_number_square_6x6():
    N = 6
    A = np.zeros((N,N))
    for j in range(N):
        for i in range(N):
            A[j,i] = j*N+i
        print(A)

# 6x6 숫자 사각형 출력
print_number_square_6x6()