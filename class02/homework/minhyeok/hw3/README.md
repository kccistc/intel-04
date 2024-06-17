# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 163
./splitted_dataset/train: 130
./splitted_dataset/train/good: 64
./splitted_dataset/train/bad: 66
./splitted_dataset/val: 33
./splitted_dataset/val/good: 16
./splitted_dataset/val/bad: 17
```

## Training 결과
|----|----|----|----|----|----|----|
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|EfficientNet-V2-S| 1.0 | 14.21 | '0:01:18.647442' | 64 | 0.0076 | ---- |
|EfficientNet-B0| 1.0 | 78.16 | '0:00:18.168751' | 64 | 0.0051 | ---- |
|DeiT-Tiny| 1.0 | 90.22 | '0:00:15.901222' | 64 | 0.0002 | ---- |
|MobileNet-V3-large-1x| 1.0 | 122.01 | '0:00:10.601123' | 64 | 0.0058 | ---- |


## FPS 측정 방법
```
inference_start_time = time.time()
inference_time = time.time() - inference_start_time fps = 1 / inference_time log.info(f'FPS : {fps:.2f}')
```
