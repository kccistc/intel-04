# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: train, val
./splitted_dataset/train: bad, good
./splitted_dataset/train/bad: 22
./splitted_dataset/train/good: 25
./splitted_dataset/val: bad, good
./splitted_dataset/train/bad: 6
./splitted_dataset/train/good: 6
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S|1.0000|3.595182|0:00:16.904305|47|0.0071|
|EfficientNet-B0|1.0000|6.753106|0:00:11.744756|47|0.0049|
|DeiT-Tiny|1.0000|11.256190|0:00:09.615638|47|0.0001|
|MobileNet-V3-large-1x|0.83333|13.131976|0:00:10.970021|47|0.0007|


## FPS 측정 방법
FPS : Frames Per Second

frame_time = curTime - prevTime
fps = 1 / frame_time
