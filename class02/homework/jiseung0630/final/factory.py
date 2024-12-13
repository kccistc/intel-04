#!/usr/bin/env python3
"""
factory.py - 이 모듈은 공장의 영상 처리 및 제어를 위한 스크립트입니다.
"""

import sys
import threading
from argparse import ArgumentParser
from queue import Queue, Empty
from time import sleep

from cv2 import cv2
import numpy as np
from openvino.runtime import Core

from iotdemo import FactoryController
from iotdemo.motion.motion_detector import MotionDetector
from iotdemo.color.color_detector import ColorDetector

FORCE_STOP = False

def load_openvino_model(model_xml):
    """
    OpenVINO 모델을 로드하고 컴파일합니다.
    """
    core = Core()
    model = core.read_model(model=model_xml)
    compiled_model = core.compile_model(model=model, device_name='CPU')
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    return compiled_model, input_layer, output_layer

def thread_cam1(queue, compiled_model, input_layer, output_layer):
    """
    첫 번째 카메라 스레드를 실행합니다. 비디오 스트림을 읽고 모션을 감지하여 큐에 결과를 추가합니다.
    """
    motion_detector = MotionDetector()
    motion_detector.load_preset("motion.cfg", "default")

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

        queue.put(("VIDEO:Cam1 live", frame))

        detected = motion_detector.detect(frame)
        if detected is None:
            continue

        queue.put(("VIDEO:Cam1 detected", detected))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        reshaped = detected[:, :, [2, 1, 0]]
        np_data = np.moveaxis(reshaped, -1, 0)
        preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        batch_tensor = np.stack(preprocessed_numpy, axis=0)

        result = compiled_model([batch_tensor])[output_layer]
        x_ratio = result[0][1]
        circle_ratio = result[0][0]
        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        if x_ratio > circle_ratio:
            queue.put(("PUSH", "1"))

    cap.release()
    queue.put(('DONE', None))
    sys.exit(1)

def thread_cam2(queue):
    """
    두 번째 카메라 스레드를 실행합니다. 비디오 스트림을 읽고 모션을 감지하여 큐에 결과를 추가합니다.
    """
    motion_detector = MotionDetector()
    motion_detector.load_preset("motion.cfg", "default")

    color_detector = ColorDetector()
    color_detector.load_preset("color.cfg", "default")

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

        queue.put(("VIDEO:Cam2 live", frame))

        detected = motion_detector.detect(frame)
        if detected is None:
            continue

        queue.put(("VIDEO:Cam2 detected", detected))

        predict = color_detector.detect(detected)
        if not predict or len(predict) == 0:
            continue

        name, ratio = predict[0]
        ratio = ratio * 100
        print(f"{name}: {ratio:.2f}%")

        if name == 'blue':
            queue.put(("PUSH", "2"))

    cap2.release()
    queue.put(('DONE', None))
    sys.exit(1)

def imshow(title, frame, pos=None):
    """
    화면에 표출합니다.
    """
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)

def main():
    """
    메인 함수입니다.
    """
    global FORCE_STOP

    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()

    queue = Queue()
    model_xml = '../homework/jiseung0630/hw3/O_X_Classfication_EfficientNet-V2-S_2/openvino.xml'
    compiled_model, input_layer, output_layer = load_openvino_model(model_xml)

    thread1 = threading.Thread(target=thread_cam1, args=(queue, compiled_model, input_layer, output_layer))
    thread2 = threading.Thread(target=thread_cam2, args=(queue,))
    thread1.start()
    thread2.start()

    with FactoryController(args.device) as controller:
        controller.system_start()
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            try:
                name, data = queue.get(timeout=1)
            except Empty:
                continue

            if name == "VIDEO:Cam1 live":
                imshow('Cam1 live', data, (0, 0))
            elif name == "VIDEO:Cam2 live":
                imshow('Cam2 live', data, (640, 0))
            elif name == "VIDEO:Cam1 detected":
                imshow('Cam1 detected', data)
            elif name == "VIDEO:Cam2 detected":
                imshow('Cam2 detected', data)
            elif name == "PUSH":
                if data == "1":
                    controller.red = controller.red
                elif data == "2":
                    controller.orange = controller.orange
            elif name == 'DONE':
                FORCE_STOP = True
                queue.task_done()

    thread1.join()
    thread2.join()
    controller.close()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)

