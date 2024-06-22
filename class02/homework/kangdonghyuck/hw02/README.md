# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/:  59
./splitted_dataset/train: 47
./splitted_dataset/train/<bad>: 22
./splitted_dataset/train/<good>: 25
./splitted_dataset/val: 12
./splitted_dataset/val/<bad>: 6
./splitted_dataset/val/<good>: 6
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| |
|EfficientNet-B0| 
|DeiT-Tiny|
|MobileNet-V3-large-1x|1.0 |8||0:00:06.579616||64|0.0058|-|

## FPS 측정 방법

