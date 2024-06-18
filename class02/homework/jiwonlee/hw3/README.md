# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 132
./splitted_dataset/train: 105
./splitted_dataset/train/bad: 52​
./splitted_dataset/train/good: 53​
./splitted_dataset/val: 27
./splitted_dataset/train/bad: 13
./splitted_dataset/train/good: 14​
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S|1.0|12.98|00:00:33.037735|64|0.0058|-|
|EfficientNet-B0|1.0|16.36|0:00:13.785710|64|0.0049|-|
|DeiT-Tiny|1.0|13.82|0:00:10.518227|64|0.0001|-|
|MobileNet-V3-large-1x|1.0|17.20|0:00:08.427059|64|0.0058|-|


## FPS 측정 방법
deploy/python/demo.py에서 inference하는데 걸리는 시간으로 FPS 측정
```
# FPS
start_time = time.time()

# create inferencer and run
demo = inferencer(models, visualizer)
demo.run(args.input, args.loop and not args.no_show)

end_time = time.time()
elapsed_time = end_time - start_time
fps = 1 / elapsed_time
print(f"Inference FPS: {fps:.2f}")
```
