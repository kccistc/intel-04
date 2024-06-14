# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
./splitted_dataset/: 59
./splitted_dataset/val: 12
./splitted_dataset/val/bad: 5
./splitted_dataset/val/good: 7
./splitted_dataset/train: 47
./splitted_dataset/train/bad: 23
./splitted_dataset/train/good: 24
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| |
|EfficientNet-B0|1.0|94.9570675358|0:00:10.531075|64|0.0049|----|----|
|DeiT-Tiny| 
|MobileNet-V3-large-1x|1.0|146.049855287|0:00:06.846977|64|0.0058|----|----|


## FPS 측정 방법
FPS = 1000 / frame-time => 1000 / Training-time 공식으로 FPS를 측정
