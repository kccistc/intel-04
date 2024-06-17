# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
./splitted_dataset/: 65
./splitted_dataset/val: 14
./splitted_dataset/val/bad: 7
./splitted_dataset/val/good: 7
./splitted_dataset/train : 49
./splitted_dataset/train/bad: 23
./splitted_dataset/train/good: 24
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| |
|EfficientNet-B0|
|DeiT-Tiny| 
|MobileNet-V3-large-1x|1.0|1.06|0:00:06.846977|64|0.0058|----|----|


## FPS 측정 방법

65 / 61 = 1.06
