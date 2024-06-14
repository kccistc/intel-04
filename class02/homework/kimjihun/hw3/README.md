# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 101
./splitted_dataset/val: 21​
./splitted_dataset/val/bad: 11
./splitted_dataset/val/good: 10
./splitted_dataset/val: 80
./splitted_dataset/train/bad: 40​
./splitted_dataset/train/good: 40
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S|1.0|3.338|299.499794|64|0.0071|
|EfficientNet-B0|1.0|36.478|27.413640|64|0.0049|
|DeiT-Tiny|1.0|98.413|10.161299|64|0.0001|
|MobileNet-V3-large-1x|1.0|130.476|7.664220|64|0.0058|


## FPS 측정 방법
FPS = 1000 / TRAINING IME
