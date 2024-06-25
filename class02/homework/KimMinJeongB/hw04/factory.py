"""
Factory monitoring and control system using OpenVINO, motion detection, and color detection.
"""

import os
import sys
import threading
from argparse import ArgumentParser
from queue import Queue
from time import sleep

from cv2 import cv2
import numpy as np
from openvino.runtime import Core

from iotdemo import FactoryController
from iotdemo.motion.motion_detector import MotionDetector
from iotdemo.color.color_detector import ColorDetector

FORCE_STOP = False


def thread_cam1(q):
    """
    Thread function for camera 1.
    Uses MotionDetector to detect motion in the video.
    """
    motion_detector = MotionDetector()
    motion_detector.load_preset("motion.cfg", "default")

    core = Core()
    model_xml = '../homework/KimMinJeongB/hw03/MobileNet-V3-large-1x/openvino.xml'
    model = core.read_model(model=model_xml)
    compiled_model = core.compile_model(model=model, device_name='CPU')
    output_layer = compiled_model.output(0)

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

        q.put(("VIDEO:Cam1 live", frame))

        detected = motion_detector.detect(frame)
        if detected is None:
            continue

        q.put(("VIDEO:Cam1 detected", detected))

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
            q.put(("PUSH", "1"))

    cap.release()
    q.put(('DONE', None))
    sys.exit()


def thread_cam2(q):
    """
    Thread function for camera 2.
    Uses MotionDetector and ColorDetector to detect motion and color in the video.
    """
    det = MotionDetector()
    det.load_preset("./resources/motion.cfg", "default")

    color = ColorDetector()
    color.load_preset("./resources/color.cfg", "default")

    cap = cv2.VideoCapture('./resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        q.put(("VIDEO:Cam2 live", frame))

        detected = det.detect(frame=frame)

        if detected is None:
            continue

        q.put(("VIDEO:Cam2 detected", detected))

        predict = color.detect(detected)

        name, ratio = predict[0]
        ratio = ratio * 100
        print(f"{name}: {ratio:.2f}%")

        q.put(("PUSH", 2))

    cap.release()
    q.put(('DONE', None))
    sys.exit()


def imshow(title, frame, pos=None):
    """
    이미지를 윈도우에 보여줍니다.
    """
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

    q = Queue()

    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            event = q.get()
            name, data = event

            if "VIDEO:" in name:
                imshow(name[6:], data)
            elif name == "PUSH":
                ctrl.push_actuator(data)
            elif name == 'DONE':
                FORCE_STOP = True

            q.task_done()

    t1.join()
    t2.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit(1)
