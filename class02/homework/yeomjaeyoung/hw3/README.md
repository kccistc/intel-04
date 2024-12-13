# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 50
./splitted_dataset/train: 40
./splitted_dataset/train/blue: 20​
./splitted_dataset/train/white: 20
./splitted_dataset/val: 10
./splitted_dataset/val/blue: 5
./splitted_dataset/val/white: 5
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| 1.0 |14.760560|0:00:14.760560|64|0.0071|-|
|EfficientNet-B0| 1.0 |119.705042|0:00:08.353867|64|0.0049|-|
|DeiT-Tiny| 1.0 |8.899520|0:00:08.899520|64|0.0001|-|
|MobileNet-V3-large-1x|1.0 |148.9171342|0:00:06.715144|64|0.0058|-|

## FPS 측정 방법
inference_time = time.time() - inference_start_time fps = 1 / inference_time log.info(f'FPS : {fps:.2f}')
