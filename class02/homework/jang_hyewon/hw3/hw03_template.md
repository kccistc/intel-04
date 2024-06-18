# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: train, test 2
./splitted_dataset/train: good, bad 2
./splitted_dataset/train/<class#>: 18
./splitted_dataset/train/<class#>: 16
./splitted_dataset/val: good, bad 2
./splitted_dataset/val/<class#>: 4
./splitted_dataset/val/<class#>: 5
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S|1.0|13.747489314838608|0:00:21.498743|64|0.0071|-|
|EfficientNet-B0|1.0|17.25064263651656|0:00:09.290243|64|0.0049|-|
|DeiT-Tiny|1.0|14.154834703492218|0:00:07.751240|64|0.0001|-|
|MobileNet-V3-large-1x|0.9073569482288828|18.00254094701782|0:00:42.072232|64|0.0058|-|


## FPS 측정 방법

코드 뜯어보다 보니까 demo.py에서 

```
    # create inferencer and run
    demo = inferencer(models, visualizer)
    demo.run(args.input, args.loop and not args.no_show)
```
이렇게 있어서 run 부분이 추정하는 부분인거 같아, 해당 부분의 시간을 측정해서 계산하면 될거 같았습니다. 

```
    # create inferencer and run
    demo = inferencer(models, visualizer)
    
    bef_time = time.time()
    demo.run(args.input, args.loop and not args.no_show)
    aft_time = time.time()
    
    print(1 / (aft_time - bef_time))
```

그래서 이렇게 넣어 주고 계산했습니다. 
