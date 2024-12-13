# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 254
./splitted_dataset/train: 203
./splitted_dataset/train/<O>: 101​
./splitted_dataset/train/<X>: 102
./splitted_dataset/val: 51
./splitted_dataset/train/<O>: 26
./splitted_dataset/train/<X>: 25
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S|1.0|  |0:00:28.156916|64|0.0071|
|MobileNet-V3-large-1x|1.0|  | 0:00:09.828827|64|0.0058|


## FPS 측정 방법

