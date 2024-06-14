# Homework 3

Smart factory 불량 분류모델 training 결과

## Dataset 구조

```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 65
./splitted_dataset/train: 52
./splitted_dataset/train/good: 24
./splitted_dataset/train/bad: 28
./splitted_dataset/val: 13
./splitted_dataset/val/good: 6
./splitted_dataset/train/bad: 7
```

## Training 결과

| Classification model | Accuracy | FPS | Training time | Batch size | Learning rate | Other hyper-prams |
| --- | --- | --- | --- | --- | --- | --- |
| EfficientNet-V2-S | 1.0 | 47.98 | 0:00:25.657157 | 64 | 0.0071 |  |
| EfficientNet-B0 | 1.0 | 99.61 | 0:00:11.463533 | 64 | 0.0049 |  |
| DeiT-Tiny | 1.0 | 65.60 | 0:00:08.609463 | 64 | 0.0001 |  |
| MobileNet-V3-large-1x | 1.0 | 133.54 | 0:00:06.959555 | 64 | 0.0058 |  |

## FPS 측정 방법

각 모델 deploy로 생성된 [`demo.py`](http://demo.py) 수정

```python
...
import time

...

def main():
	...
	start_time = time.time()  # 추론 시작 시간
	demo.run(args.input, args.loop and not args.no_show)
	end_time = time.time()    # 추론 종료 시간
	print("Number of images: 65")
	print(f"Elapsed time: {end_time - start_time:.2f} seconds")
	print(f"FPS: {65 / (end_time - start_time):.2f}")
```

추론에 사용한 이미지 개수 65를 걸린 시간으로 나누어서 초당 처리 개수를 계산.
