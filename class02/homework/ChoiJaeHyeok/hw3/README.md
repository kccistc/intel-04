# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 100
./splitted_dataset/train: 80​
./splitted_dataset/train/bad: 40​
./splitted_dataset/train/good: 40​
./splitted_dataset/val: 20
./splitted_dataset/train/bad: 10
./splitted_dataset/train/good: 10​
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S|1.0|202.015531198|0:00:36.763364|64|0.0071||
|EfficientNet-B0|1.0|382.915717559|0:00:12.984530|64|0.0049||
|DeiT-Tiny|1.0|388.38209344|0:00:10.123262|64|0.0001||
|MobileNet-V3-large-1x|0.85|484.058595104|0:00:07.621356|64|0.0058||


## FPS 측정 방법
모델 optimize 진행 후 나오는 Avg time per image을 1초에 대해 나누기 진행

```
(.otx) ubuntu@ubuntu-500TFA-500SFA:~/hw3/otx_DeiTTiny$ otx optimize --load-weights openvino_model/openvino.xml --output pot_model

...
2024-06-14 18:34:26,368 | INFO : Avg time per image: 0.0025747839998075507 secs
2024-06-14 18:34:26,368 | INFO : Total time: 0.05149567999615101 secs
2024-06-14 18:34:26,369 | INFO : Classification OpenVINO inference completed
Performance(score: 1.0, dashboard: (3 metric groups))
```
