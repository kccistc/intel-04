# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 81
./splitted_dataset/train: 64
./splitted_dataset/train/<bad>: 24
./splitted_dataset/train/<good>: 40
./splitted_dataset/val: 17
./splitted_dataset/val/<bad>: 7
./splitted_dataset/val/<good>: 10​
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| |1.0|8.333333|0:02:00.733791|64|0.0071|----|
|EfficientNet-B0| |1.0|81.5294|0:00:12.265528|64|0.0049|----|
|DeiT-Tiny| |1.0|107.25585|0:00:09.323528|64|0.0001|----|
|MobileNet-V3-large-1x| |1.0|132.100396301|0:00:07.569413|64|0.0058|----|


## FPS 측정 방법
