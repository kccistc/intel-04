# Homework03

Smart factory 불량 분류모델 training 결과

## Dataset 구조

```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 59
./splitted_dataset/train: 47
./splitted_dataset/train/<class#>: 22
./splitted_dataset/train/<class#>: 25
./splitted_dataset/val: 12
./splitted_dataset/train/<class#>: 6
./splitted_dataset/train/<class#>: 6

```

## Training 결과

| Classification model | Accuracy | FPS | Training time | Batch size | Learning rate | Other hyper-prams |
| --- | --- | --- | --- | --- | --- | --- |
| EfficientNet-V2-S | 1.0 |  | 0:00:28.887994 | 64 | 7.100e-03 |  |
| MobileNet-V3-large-1x | 1.0 |  | 0:00:17.129582 | 64 | 1.0e-05 |  |

## FPS 측정 방법
