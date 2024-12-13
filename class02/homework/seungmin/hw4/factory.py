#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

from cv2 import cv2
import numpy as np
from openvino.runtime import Core

from iotdemo import FactoryController
from iotdemo.motion.motion_detector import MotionDetector
from iotdemo.color.color_detector import ColorDetector

FORCE_STOP = False

def thread_cam1(q):
    # TODO: MotionDetector
    motion_detector = MotionDetector()
    motion_detector.load_preset("motion.cfg", "default")
    # TODO: Load and initialize OpenVINO
    ie = Core()
    model_xml = '../homework/seungmin/hw3/DeiT-Tiny/model.xml'
    #model_bin = '../homework/suhwanjo/hw3/MobileNet-V3-large-1x/model.bin'
    model = ie.read_model(model=model_xml)
    compiled_model = ie.compile_model(model=model, device_name='CPU')
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # TODO: HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    input_video = './resources/conveyor.mp4'
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video}")
        return

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live",frame))
        # TODO: Motion detect
        detected = motion_detector.detect(frame)
        if detected is None:
            continue

        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(("VIDEO:Cam1 detected",detected))
        # abnormal detect
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        reshaped = detected[:, :, [2, 1, 0]]
        np_data = np.moveaxis(reshaped, -1, 0)
        preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        batch_tensor = np.stack(preprocessed_numpy, axis=0)

        # TODO: Inference OpenVINO
        result = compiled_model([batch_tensor])[output_layer]
        x_ratio = result[0][1]  
        circle_ratio = result[0][0] 
        # TODO: Calculate ratios
        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # TODO: in queue for moving the actuator 1
        # 불량품인 경우 actuator 1
        if x_ratio > circle_ratio:
            q.put(("PUSH", "1"))
        
    cap.release()
    q.put(('DONE', None))
    exit()

def thread_cam2(q):
    # TODO: MotionDetector
    motion_detector = MotionDetector()
    motion_detector.load_preset("motion.cfg","default")

    # TODO: ColorDetector
    color_detector = ColorDetector()
    color_detector.load_preset("color.cfg","default")

    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    input_video2 = './resources/conveyor.mp4'
    cap2 = cv2.VideoCapture(input_video2)
    if not cap2.isOpened():
        print(f"Error: Could not open video {input_video2}")
        return

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap2.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live", frame))
        # TODO: Detect motion
        detected = motion_detector.detect(frame)
        if detected is None:
            continue
        # # TODO: Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(("VIDEO:Cam2 detected", detected))
        # TODO: Detect color
        predict = color_detector.detect(detected)
        name, ratio = predict[0]
        ratio = ratio * 100
        # TODO: Compute ratio
#        print(f"{name}: {ratio:.2f}%")

        # TODO: Enqueue to handle actuator 2
        # 파란색인 경우 actuator 2
        if name == 'blue':
            q.put(("PUSH","2"))

    cap2.release()
    q.put(('DONE', None))
    exit()

def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)

def main():
    global FORCE_STOP

    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()

    # TODO: HW2 Create a Queue
    q = Queue()
    # TODO: HW2 Create thread_cam1 and thread_cam2 threads and start them.
    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()
    with FactoryController(args.device) as ctrl:
        ctrl.system_start()
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # TODO: HW2 get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            # 큐에서 아이템 가져오기
            name, data = q.get()

            # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            if name == "VIDEO:Cam1 live":
                imshow('Cam1 live', data, (0,0))
            elif name == "VIDEO:Cam2 live":
                imshow('Cam2 live', data, (640,0))
            elif name == "VIDEO:Cam1 detected":
                imshow('Cam1 detected',data)
            elif name == "VIDEO:Cam2 detected":
                imshow('Cam2 detected',data)
            # TODO: Control actuator, name == 'PUSH'
            elif name == "PUSH":
                if data == "1":
                    ctrl.red = ctrl.red
                elif data == "2":
                    ctrl.orange = ctrl.orange
            elif name == 'DONE':
                FORCE_STOP = True
                q.task_done()
    
    t1.join()
    t2.join()
    ctrl.close()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()