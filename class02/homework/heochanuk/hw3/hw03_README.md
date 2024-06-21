# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
find ./splitted_dataset/train/ -name "*jpg" | wc -l

./splitted_dataset/: 48
./splitted_dataset/train: 38​
./splitted_dataset/train/<class#>: 20​
./splitted_dataset/train/<class#>: 18​
./splitted_dataset/val: 10
./splitted_dataset/train/<class#>: 6​
./splitted_dataset/train/<class#>: 4​
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| 	1.0 	|	|0:01:18.817406 	|64 	|0.00355 	|----
|EfficientNet-B0| 	1.0 	|	|0:00:18.494986 	|64 	|0.00245 	|----
|DeiT-Tiny 	|1.0 	|	|0:00:16.258080 	|64 	|5e-05 	|----
|MobileNet-V3-large-1x 	|1.0 	|	|0:00:13.059633 	|64 	|2.900e-03 	|----
