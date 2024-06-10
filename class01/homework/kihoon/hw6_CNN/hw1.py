import cv2
import numpy as np

def apply_filter(img, kernel):
    return cv2.filter2D(img, -1, kernel)

def show_combined_image(images):
    combined = cv2.hconcat(images)
    cv2.imshow('edge', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('123.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: 이미지 파일을 찾을 수 없습니다. 파일 경로를 확인하세요.")
    exit()

images = []

# Identity
kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
images.append(apply_filter(img, kernel))

# Ridge or edge detection
kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
images.append(apply_filter(img, kernel))

kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
images.append(apply_filter(img, kernel))

# Sharpen
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
images.append(apply_filter(img, kernel))

# Box blur
kernel = np.array(1/9 * np.ones((3, 3)))
images.append(apply_filter(img, kernel))

# Gaussian blur 3x3
kernel = np.array(1/16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]))
images.append(apply_filter(img, kernel))

# Gaussian blur 5x5
kernel = np.array(1/256 * np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]))
images.append(apply_filter(img, kernel))

show_combined_image(images)