# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 59
./splitted_dataset/train: 47
./splitted_dataset/train/bad: 22
./splitted_dataset/train/good: 25
./splitted_dataset/val: 12
./splitted_dataset/train/<class#>: 6
./splitted_datase
t/train/<class#>: 6
```
## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|
|MobileNet-V3-large-1x|1.0 ||213.68|0:00:06.715144||64|0.0046799|-|

## FPS 측정 방법

1/Avg time per image
