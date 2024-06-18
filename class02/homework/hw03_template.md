# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 108
./splitted_dataset/train: 86
./splitted_dataset/train/<OK>: 44​
./splitted_dataset/train/<NOT_OK>: 42​
./splitted_dataset/val: 22
./splitted_dataset/train/<OK>: 10​
./splitted_dataset/train/<NOT_OK>: 12
```

## Training 결과
|Classification model|Accuracy|FPS   |Training time   |Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S  |    1.0  |30/50 | 0:01:18.817406 | 64       |   0.00355   |        
|EfficientNet-B0    |    1.0  |	X/150| 0:00:18.494986 |	64 	 |   0.00245   |
|DeiT-Tiny	    |    1.0  |X /52 | 0:00:16.258080 |64 	 |    5e-05    |
|MobileNet-V3-large-1x|  1.0  |200/215|0:00:13.059633 |	64 	 |  2.900e-03  |


## FPS 측정 방법


 ????

