from PIL import Image
import numpy as np

# 1. 임의의 이미지 파일을 불러오기
image_path = 'images.jpeg'  # 이미지 파일 경로를 입력하세요
image = Image.open(image_path)
image_array = np.array(image)

# 2. Numpy의 expand_dims를 사용해서 이미지 파일의 차원을 하나 더 늘리기
# (Height, Width, Channel) -> (Batch, Height, Width, Channel)
image_expanded = np.expand_dims(image_array, axis=0)
print("After expand_dims:", image_expanded.shape)

# 3. Numpy의 transpose를 이용해서 차원의 순서를 변경하기
# (Batch, Height, Width, Channel) -> (Batch, Channel, Width, Height)
image_transposed = np.transpose(image_expanded, (0, 3, 2, 1))
print("After transpose:", image_transposed.shape)
