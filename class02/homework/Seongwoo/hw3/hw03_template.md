# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 36
./splitted_dataset/train: 28
./splitted_dataset/train/<class#good>: 15
./splitted_dataset/train/<class#bad>: 13
./splitted_dataset/val: 8
./splitted_dataset/train/<class#good>: 3
./splitted_dataset/train/<class#bad>: 5
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| 1.0 | 47.710069 | 20.959936 | 64 | 0.0071 | 
|EfficientNet-B0| 1.0 | 105.129975 | 09.512035 | 64 | 0.0049 | 
|DeiT-Tiny| 1.0 | 126.900461 | 07.880192 | 64 |0.0001|
|MobileNet-V3-large-1x| 1.0 | 168.477519 | 05.935551 | 64 | 0.0058 | 


## FPS 측정 방법
