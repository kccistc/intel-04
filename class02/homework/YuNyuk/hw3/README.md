# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 61
./splitted_dataset/train: 48​
./splitted_dataset/train/<bad>: 20​
./splitted_dataset/train/<good>: 28​
./splitted_dataset/val: 13
./splitted_dataset/train/<bad>: 7​
./splitted_dataset/train/<good>: 6​
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S|----|----|----|----|----|----|
|EfficientNet-B0| ----|----|----|----|----|----|
|DeiT-Tiny| ----|----|----|----|----|----|
|MobileNet-V3-large-1x|1.000|----|0:00:06.600|48|0.0058|----|


## FPS 측정 방법
fps= 1 / inference_time
