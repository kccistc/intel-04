import collections
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
from IPython import display
import openvino as ov
from openvino.tools.mo.front import tf as ov_tf_front
from openvino.tools import mo

import notebook_utils as utils
import ipywidgets as widgets
from ultralytics import YOLO

core = ov.Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value="AUTO",
    description="Device:",
    disabled=False,
)

precision = "FP16"
model_name = "ssdlite_mobilenet_v2"
# The output path for the conversion.
converted_model_path = Path("./model") / f"{model_name}_{precision.lower()}.xml"

# Read the network and corresponding weights from a file.
print(converted_model_path)


model = core.read_model(model=converted_model_path)
# Compile the model for CPU (you can choose manually CPU, GPU etc.)
# or let the engine choose the best available device (AUTO).
compiled_model = core.compile_model(model=model, device_name=device.value)

# Get the input and output nodes.
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Get the input size.
height, width = list(input_layer.shape)[1:3]


#######

# https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
classes = [
    "background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "street sign",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eye glasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "plate",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "mirror",
    "dining table",
    "window",
    "desk",
    "toilet",
    "door",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "hair brush",
]

# Colors for the classes above (Rainbow Color Map).
colors = cv2.applyColorMap(
    src=np.arange(0, 255, 255 / len(classes), dtype=np.float32).astype(np.uint8),
    colormap=cv2.COLORMAP_RAINBOW,
).squeeze()


def process_results(frame, results, thresh=0.6):
    # The size of the original frame.
    h, w = frame.shape[:2]
    # The 'results' variable is a [1, 1, 100, 7] tensor.
    results = results.squeeze()
    boxes = []
    labels = []
    scores = []
    for _, label, score, xmin, ymin, xmax, ymax in results:
        # Create a box with pixels coordinates from the box with normalized coordinates [0,1].
        boxes.append(tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h))))
        labels.append(int(label))
        scores.append(float(score))

    # Apply non-maximum suppression to get rid of many overlapping entities.
    # See https://paperswithcode.com/method/non-maximum-suppression
    # This algorithm returns indices of objects to keep.
    indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.6)

    # If there are no boxes.
    if len(indices) == 0:
        return []

    # Filter detected objects.
    return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]


def draw_boxes(frame, boxes):
    for label, score, box in boxes:
        # Choose color for the label.
        color = tuple(map(int, colors[label]))
        
        if(label == 1):
            # Draw a box.
            x2 = box[0] + box[2]
            y2 = box[1] + box[3]
            cv2.rectangle(img=frame, pt1=box[:2], pt2=(x2, y2), color=color, thickness=3)
    
            # Draw a label name inside the box.
            cv2.putText(
                img=frame,
                text=f"{classes[label]} {score:.2f}",
                org=(box[0] + 10, box[1] + 30),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=frame.shape[1] / 1000,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    return frame

# Main processing function to run object detection.
def run_object_detection(frame):
    if frame is None:
        print("Source ended")
        return
    # If the frame is larger than full HD, reduce size to improve the performance.
    scale = 1280 / max(frame.shape)
    if scale < 1:
        frame = cv2.resize(
            src=frame,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        
    # Resize the image and change dims to fit neural network input.
    input_img = cv2.resize(src=frame, dsize=(width, height), interpolation=cv2.INTER_AREA)
    # Create a batch of images (size = 1).
    input_img = input_img[np.newaxis, ...]

    # Measure processing time.

    start_time = time.time()
    # Get the results.
    results = compiled_model([input_img])[output_layer]
    stop_time = time.time()
    # Get poses from network results.
    boxes = process_results(frame=frame, results=results)

    # Draw boxes on a frame.
    frame = draw_boxes(frame=frame, boxes=boxes)

    _, f_width = frame.shape[:2]

    return frame
    
    # Mean processing time [ms].
#########################################################################################
# segmentation 관련 코드

# 학습된 모델을 로드합니다.
model2 = YOLO('crack_trace.pt')

def crack_segmentation(frame):
    # 모델을 사용하여 프레임에서 객체 감지
    results = model2(frame)

    # 결과를 프레임에 그리기
    annotated_frame = results[0].plot()

    # 화면에 표시
    return annotated_frame

#########################################################################################
# 카메라 2개 관련 코드 


import cv2

def initializecamera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Camera {index} could not be opened.")
        return None
    return cap

def captureframe(cap):
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        return None
    return frame

def display_cameras():
    cap1 = initializecamera(4)  # 첫 번째 카메라 (0번 장치)
    cap2 = initializecamera(7)  # 두 번째 카메라 (1번 장치)

    if not cap1 or not cap2:
        print("Ensure both cameras are connected and try again.")
        return

    while True:
        frame1 = captureframe(cap1)
        frame2 = captureframe(cap2)

        if frame1 is None or frame2 is None:
            break

        # 모델을 통한 처리 
        frame1 = run_object_detection(frame1)
        frame2 = crack_segmentation(frame2)

        # 이미지 크기와 타입을 일치시킵니다.
        frame2_processed_resized = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        # 두 프레임을 세로로 결합하여 하나의 화면에 두 카메라 화면을 출력합니다.
        combined_frame = cv2.vconcat([frame1, frame2_processed_resized])

        # 결합된 프레임 출력

        cv2.imshow('Combined Camera View', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
  
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


#########################################################################################
# main

display_cameras()