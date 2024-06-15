# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/:65
./splitted_dataset/train: 52​
./splitted_dataset/train/<bad>: 27
./splitted_dataset/train/<good>:25
./splitted_dataset/val: 13
./splitted_dataset/train/<bad>: 6​
./splitted_dataset/train/<good>:7​
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S|1.0000 |39.6089|25.246828|64|0.0071
|EfficientNet-B0| 1.0000|90.018|11.108877|64|0.0049
|DeiT-Tiny|1.0000|109.877|9.101034|64|0.0001 
|MobileNet-V3-large-1x|0.8571 |119.463 |8.370792 |64|0.0058


## FPS 측정 방법
