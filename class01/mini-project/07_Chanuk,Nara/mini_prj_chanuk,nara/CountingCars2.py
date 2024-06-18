import cv2
import time
import collections
import numpy as np
from ultralytics import YOLO
from ultralytics.solutions import ObjectCounter
import torch
import openvino as ov

# Define video source
VIDEO_SOURCE = "./test_video4.mp4"
# Define the path to the OpenVINO model
DET_MODEL_NAME = "yolov8n"
models_dir = "./models"
det_model_path = f"{models_dir}/{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"

# Load the YOLO model
try:
    det_model = YOLO(f"{models_dir}/{DET_MODEL_NAME}.pt")
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    exit(1)

# Initialize OpenVINO core
core = ov.Core()

# Load the model
try:
    det_ov_model = core.read_model(det_model_path)
    compiled_model = core.compile_model(det_ov_model, "CPU")
    print("OpenVINO model compiled successfully.")
except Exception as e:
    print(f"Failed to load or compile OpenVINO model: {e}")
    exit(1)

def infer(*args):
    result = compiled_model(args)
    return torch.from_numpy(result[0])

# Check if the model has a predictor and initialize it
try:
    predictor = det_model.model
    if predictor is not None:
        predictor.inference = infer
        predictor.model.pt = False
        print("YOLO model predictor initialized successfully.")
    else:
        raise RuntimeError("Failed to initialize the YOLO model predictor.")
except AttributeError as e:
    print(f"Attribute error: {e}")
    exit(1)

def run_inference(source):
    try:
        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), "Error reading video file"

        # Get video frame dimensions
        ret, frame = cap.read()
        if not ret:
            print("Failed to read the video file.")
            exit(1)
            
        height, width, _ = frame.shape
        print(height, width)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the first frame

        # Set line points at the bottom of the frame
        line_points = [(0, height - 40), (2 * width, height - 40)]  # Line or region points 10 pixels from the bottom
        classes_to_count = [2, 3, 5, 7]  # COCO dataset classes to count (e.g., car, bus, truck, etc.)

        # Init Object Counter
        counter = ObjectCounter(
            view_img=False, 
            reg_pts=line_points, 
            classes_names=det_model.names, 
            draw_tracks=False, 
            line_thickness=2, 
            line_dist_thresh=75,  # Increased sensitivity for counting
            view_in_counts=False, 
            view_out_counts=False
        )
        # Processing time
        processing_times = collections.deque(maxlen=200)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            start_time = time.time()
            tracks = det_model.track(frame, persist=True, show=False, classes=classes_to_count)
            frame = counter.start_counting(frame, tracks)
            stop_time = time.time()

            processing_times.append(stop_time - start_time)

            # Mean processing time [ms].
            _, f_width = frame.shape[:2]
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=f_width / 1000,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            # Get the counts. Counts are getting as 'OUT'
            counts = counter.out_counts

            # Define the text to display
            text = f"Count: {counts}"
            fontFace = cv2.FONT_HERSHEY_COMPLEX
            fontScale = 0.75  # Adjust scale as needed
            thickness = 2

            # Calculate the size of the text box
            (text_width, text_height), _ = cv2.getTextSize(text, fontFace, fontScale, thickness)

            # Define the upper right corner for the text
            top_right_corner = (frame.shape[1] - text_width - 20, 40)
            # Draw the count of "OUT" on the frame
            cv2.putText(
                img=frame,
                text=text,
                org=(top_right_corner[0], top_right_corner[1]),
                fontFace=fontFace,
                fontScale=fontScale,
                color=(0, 0, 255),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

            # Show the frame
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Print the total counts for each class when the video ends
        print("Total counts per class:")
        for class_name, count_info in counter.class_wise_count.items():
            total_count = count_info["IN"] + count_info["OUT"]
            print(f"{class_name}: {total_count}")

    except KeyboardInterrupt:
        print("Interrupted")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference(VIDEO_SOURCE)