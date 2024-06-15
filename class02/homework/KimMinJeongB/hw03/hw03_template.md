# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 97
./splitted_dataset/train: 77
./splitted_dataset/train/<bad>: 37​
./splitted_dataset/train/<good>: 40
./splitted_dataset/val: 20
./splitted_dataset/val/<bad>: 11
./splitted_dataset/val/<good>: ​9
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| 1.0 | 2.90 | 0:00:26.545214 | 64 | 0.0071 |
|EfficientNet-B0| 1.0 | 7.75 | 0:00:09.935971 | 64 | 0.0049 |
|DeiT-Tiny| 1.0 | 7.47 | 0:00:10.306162 | 64 | 0.0001 |
|MobileNet-V3-large-1x| 1.0 | 10.55 | 0:00:07.307438 | 64 | 0.0058 |


## FPS 측정 방법
![FPS](https://cdn.discordapp.com/attachments/1247775254416326708/1251120318387322980/Screenshot_from_2024-06-14_19-24-23.png?ex=666d6c18&is=666c1a98&hm=03db6ca165768f542c38f9a706a2a49085dd9dcf3615085b5d8e80f830dad74b&)

* Number of frames processed : 주어진 시간 동안 처리된 프레임의 수를 의미, 훈련 데이터의 이미지 수<br>
* Total time taken : 주어진 시간 동안 총 걸린 시간 의미
