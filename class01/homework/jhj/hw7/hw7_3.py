import cv2
import numpy as np

# 이미지 파일 경로


# 이미지를 불러오기 (흑백 모드)
img = cv2.imread('/home/ubuntu/Luicd2.webp', cv2.IMREAD_GRAYSCALE)

# 이미지가 제대로 불러와졌는지 확인
if img is None:
    print(f"Error: Unable to load image at path: {image_path}")
else:
    # 커널 정의
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    print(kernel)
    
    # 필터 적용
    output = cv2.filter2D(img, -1, kernel)
    
    # 결과 출력
    cv2.imshow('edge', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
