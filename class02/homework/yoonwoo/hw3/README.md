# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 117
./splitted_dataset/train: 93​
./splitted_dataset/train/<Good>: 51​
./splitted_dataset/train/<Bad>: 42​
./splitted_dataset/val: 24
./splitted_dataset/val/<Good>: 12​
./splitted_dataset/val/<Bad>: 12​
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|EfficientNet-V2-S| 1.0| 125| 0:00:08.279238| 64| 0.0071
|EfficientNet-B0| 1.0| 125| 0:00:08.279238| 64| 0.0049|
|DeiT-Tiny| 1.0| 125| 0:00:08.279238| 64| 0.0001|
|MobileNet-V3-large-1x| 1.0| 142|'0:00:07.755650'| 64|0.0058|


## FPS 측정 방법
1000 / frameTime
