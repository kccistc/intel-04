import os

import logging as log
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

import collections
import tarfile
import time
from pathlib import Path

#from IPython import display
import openvino as ov
from openvino.tools.mo.front import tf as ov_tf_front
from openvino.tools import mo

#########################################
########### declare ###################
########################################

core = ov.Core()
device = 'CPU'

model_path_face = "./intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml"
model_face = core.read_model(model=model_path_face)
compiled_model_face = core.compile_model(model=model_face, device_name=device)
model_path_emotion = "./intel/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml"
#model_path_emotion = "./models/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml"
model_emotion = core.read_model(model=model_path_emotion)
compiled_model_emo = core.compile_model(model=model_emotion, device_name=device)

# Get the input and output nodes.
input_layer = compiled_model_face.input(0)
output_layer = compiled_model_face.output(0)

input_layer_emo = compiled_model_emo.input(0)
output_layer_emo = compiled_model_emo.output(0)

# Get the input size.
height, width = list(input_layer.shape)[2:4]
h_emo, w_emo = list(input_layer_emo.shape)[2:4]
camera = cv2.VideoCapture(0)


emotion_class = ["neutral", "happy", "sad", "surprise", "anger"]

#####################################
############# Functions #############
#####################################


def process_results(frame, results, thresh=0.6):
    # The size of the original frame.
    h, w = frame.shape[:2]
    # The 'results' variable is a [1, 1, 200, 7] tensor.
    results = results.squeeze()
    boxes = []
    labels = []
    scores = []
    for _, label, score, xmin, ymin, xmax, ymax in results:
        # Create a box with pixels coordinates from the box with normalized coordinates [0,1].
        box_face = tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h)))
        boxes.append(box_face)
        labels.append(int(label))
        scores.append(float(score))
        
    indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.6)
    
    # If there are no boxes.
    if len(indices) == 0:
        return []

    # Filter detected objects.
    return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]

def draw_boxes(frame, boxes):
    for label, score, box in boxes:
        # Choose color for the label.
        # Draw a box.
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        cv2.rectangle(img=frame, pt1=box[:2], pt2=(x2, y2), color=(255,255,255), thickness=3)

    return frame


while True :
    ret, frame = camera.read()
    input_img = cv2.resize(src=frame, dsize=(width, height), interpolation=cv2.INTER_AREA)
    input_img = input_img.transpose((2, 0, 1))
    input_img = input_img[np.newaxis, ...]
    results = compiled_model_face([input_img])[output_layer]
    
    boxes = process_results(frame=frame, results=results)
    
    if len(boxes) > 0 :
        box_break = False
        for _,_,box in boxes:
            for i in range(len(box)):
                if box[i] < 0 :
                    box_break = True
                    break
                    print('box break')
            if box_break == True :
                break
            input_emo = frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            input_emo = cv2.resize(src=input_emo, dsize=(w_emo, h_emo), interpolation=cv2.INTER_AREA)
            input_emo = input_emo.transpose((2, 0, 1))
            input_emo = input_emo[np.newaxis, ...]
        
            results_emo = compiled_model_emo([input_emo])[output_layer_emo]
            results_emo = results_emo.squeeze()
            cv2.putText(
		    img=frame,
		    text=f"{emotion_class[np.argmax(results_emo)]}",
		    org=(box[0] + 10, box[1] + 30),
		    fontFace=cv2.FONT_HERSHEY_COMPLEX,
		    fontScale=frame.shape[1] / 1000,
		    color=(255,255,0),
		    thickness=1,
		    lineType=cv2.LINE_AA,
        	)
            if (results_emo[4] > 0.8) :
                #os.system("shutdown -t 30")
                print("shutdown")
                exit()
    frame = draw_boxes(frame=frame, boxes=boxes)

    cv2.imshow(winname='window', mat=frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
camera.release()
cv2.destroyAllWindows()
