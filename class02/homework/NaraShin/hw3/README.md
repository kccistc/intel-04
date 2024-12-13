# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 59
./splitted_dataset/train: 47​
./splitted_dataset/train/bad: 23​
./splitted_dataset/train/good: 24​
./splitted_dataset/val: 12
./splitted_dataset/train/bad: 5​
./splitted_dataset/train/good: 7​
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S|1.0|35|0:00:28.528895|64|0.0071|
|EfficientNet-B0|1.0|100|0:00:10.232583|64|0.0049|
|DeiT-Tiny|1.0|125|0:00:08.470946|64|0.0001|
|MobileNet-V3-large-1x|1.0|166|0:00:06.597508|64|0.0058|


## FPS 측정 방법
1000/Training time
