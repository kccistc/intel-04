# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 202
./splitted_dataset/train: 166
./splitted_dataset/train/defective: ​78
./splitted_dataset/train/normal: 88
./splitted_dataset/val: 42
./splitted_dataset/train/defective: 20
./splitted_dataset/train/normal: 22​
```

## Training 결과
|Classification model|Accuracy|FPS(CPU/GPU)|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S|1.0|63/34|0:00:44.835502|64|0.0071|Epochs: 90, Optimizer: SGD|
|EfficientNet-B0|1.0|55/34|0:00:17.029518|64|0.0049|Epochs: 90, Optimizer: SGD|
|DeiT-Tiny|1.0|54/34|0:00:12.069608|64|0.0001|Epochs: 90, Optimizer: AdamW|
|MobileNet-V3-large-1x|0.92857|520/302|0:00:10.419891|64|0.0058|Epochs:90, Optimizer:SGD|


## FPS 측정 방법
conveyor 영상으로 Inference 할 때의 평균 FPS를 측정.
```
def main():
    # Load the model
    model_path = "outputs/deploy/model/model.xml"  # 모델 파일 경로 수정
    ie = Core()
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")

    # Get input and output nodes
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # Open the video file
    input_video = "outputs/conveyor.mp4"  # 비디오 파일 경로 수정
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video}")
        return

    frame_count = 0
    total_time = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        input_image = cv2.resize(frame, (224, 224))  # Assuming the model input size is 224x224
        input_image = input_image.transpose(2, 0, 1)  # Change data layout from HWC to CHW
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
        input_image = input_image.astype(np.float32)

        start_time = time.time()
        # Perform inference
        result = compiled_model([input_image])[output_layer]
        end_time = time.time()

        total_time += (end_time - start_time)
        frame_count += 1

        # Process the result (this part may vary depending on the model's output)
        predicted_class = np.argmax(result)
        confidence = np.max(result)

        # Draw the result on the frame
        label = f"Class: {predicted_class}, Confidence: {confidence:.2f}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if frame_count > 0:
        avg_time_per_frame = total_time / frame_count
        fps = frame_count / total_time
        print(f"Average time per frame: {avg_time_per_frame:.4f} seconds")
        print(f"Average FPS: {fps:.2f}")
    else:
        print("No frames were processed.")

if __name__ == "__main__":
    main()
```
