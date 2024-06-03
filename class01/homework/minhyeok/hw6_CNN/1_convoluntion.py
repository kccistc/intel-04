import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
img = cv2.imread('hw6_CNN/dog.jpeg', cv2.IMREAD_GRAYSCALE)

# 여러 커널(마스크?!) 정의!
kernels = {
    'identity': np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]]),
    'ridge': np.array([[ 0, -1,  0],
                       [-1,  4, -1],
                       [ 0, -1,  0]]),
    'edge_detection': np.array([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]]),
    'sharpen': np.array([[ 0, -1, 0],
                         [-1,  5, -1],
                         [ 0, -1, 0]]),
    'box_blur': np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]]) * (1/9.0),
    'gaussian_blur3': np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]]) * (1/16.0),
    'gaussian_blur3': np.array([[1,  4,  6,  4, 1],
                                [4, 16, 24, 16, 4],
                                [6, 24, 36, 24, 6],
                                [4, 16, 24, 16, 4],
                                [1,  4,  6,  4, 1],]) * (1/256.0),
    'unsharp_masking': np.array([[1,  4,    6,  4, 1],
                                 [4, 16,   24, 16, 4],
                                 [6, 24, -476, 24, 6],
                                 [4, 16,   24, 16, 4],
                                 [1,  4,    6,  4, 1],]) * (-1/256.0)
}

# 결과 저장할 공간
results = {}

# # 커널 하나씩 적용
# for name, kernel in kernels.items():
#     filtered_img = cv2.filter2D(img, -1, kernel)
#     cv2.imshow(name, filtered_img)

# 위 코드는 너무 많은 페이지가 생성되기 때문에 하나의 창에 표현
# 이미지를 필터링하여 수평으로 결합하기
combined_img = None # 하나로 표현하기 위한 공간
for kernel in kernels.values():
    filtered_img = cv2.filter2D(img, -1, kernel)
    if combined_img is None:
        combined_img = filtered_img
    else:
        combined_img = np.hstack((combined_img, filtered_img)) # 합성

# 이미지 표시
cv2.imshow('Combined Images', combined_img)

# 키 입력 대기 후 이미지 창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# # 결과 출력(plt 사용)
# '''
# subplots는 여러 개의 서브 플롯을 포함하는 하나의 그림 생성하는 것.
# subplots(행의 수, 열의 수, ...)
# '''
# num_kernels = len(kernels)
# fig, axes = plt.subplots(1, num_kernels + 1, figsize=(20, 5))

# # 원본 이미지 출력
# axes[0].imshow(img, cmap='gray')
# axes[0].set_title('Original')
# axes[0].axis('off')

# # 필터링된 이미지 출력
# for ax, (name, filtered_img) in zip(axes[1:], results.items()):
#     ax.imshow(filtered_img, cmap='gray')
#     ax.set_title(name)
#     ax.axis('off')

# plt.show()