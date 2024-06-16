# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 158
./splitted_dataset/train: 126
./splitted_dataset/train/<bad>: 58
./splitted_dataset/train/<good>: 68
./splitted_dataset/val: 32
./splitted_dataset/val/<bad>: 15
./splitted_dataset/val/<good>: 17
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| |
|EfficientNet-B0| 
|DeiT-Tiny| 
|MobileNet-V3-large-1x|1.0|146|0:00:06.924788|64|0.0058|----|----|


## FPS 측정 방법
1000/Training time