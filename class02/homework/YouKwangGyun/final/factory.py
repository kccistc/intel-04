#!/usr/bin/env python3

"""
Factory tool for processing video streams from two cameras,
detecting motion, performing inference using OpenVINO, and controlling actuators.
"""

import os
import threading
from argparse import ArgumentParser
from queue import Queue
from time import sleep

import sys
import cv2
import numpy as np
from openvino.inference_engine import IECore

from iotdemo import FactoryController, MotionDetector, ColorDetector

VIDEO_PATH = "./resources/conveyor.mp4"

FORCE_STOP = False

def thread_cam1(q):
    """
    Thread function for processing camera 1 input.
    Detects motion and performs inference using OpenVINO model.
    """
    # MotionDetector
    det = MotionDetector()
    det.load_preset("./resources/motion.cfg", "default")

    # Load and initialize OpenVINO
    ie = IECore()
    compiled_model = ie.read_network(model="resources/openvino.xml",
                                                weights="resources/openvino.bin")
    model = ie.load_network(network=compiled_model, device_name='CPU')

    # Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture(VIDEO_PATH)

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live", frame))

        # Motion detect
        detected = det.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(("VIDEO:Cam1 detected", detected))

        # abnormal detect
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        reshaped = detected[:, :, [2, 1, 0]]
        np_data = np.moveaxis(reshaped, -1, 0)
        preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        batch_tensor = np.stack(preprocessed_numpy, axis=0)

        # Inference OpenVINO
        input_blob = next(iter(model.input_info))
        result = model.infer({input_blob: batch_tensor})
        predictions = next(iter(result.values()))
        probs = predictions.reshape(-1)
        softmax = np.exp(probs) / np.sum(np.exp(probs))
        x_ratio = softmax[0]
        circle_ratio = softmax[1]

        # Calculate ratios
        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # in queue for moving the actuator 1
        if x_ratio < circle_ratio:
            q.put(("PUSH", 1))

    cap.release()
    q.put(('DONE', None))
    sys.exit()


def thread_cam2(q):
    """
    Thread function for processing camera 2 input.
    Detects motion and color.
    """
    # MotionDetector
    det = MotionDetector()
    det.load_preset("./resources/motion.cfg", "default")

    # ColorDetector
    color = ColorDetector()
    color.load_preset("./resources/color.cfg", "default")

    # Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture(VIDEO_PATH)

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live", frame))

        # Detect motion
        detected = det.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(("VIDEO:Cam2 detected", detected))

        # Detect color
        predict = color.detect(detected)

        # Compute ratio
        name, ratio = predict[0]
        ratio *= 100
        print(f"{name}: {ratio:.2f}%")

        # Enqueue to handle actuator 2
        if name == 'blue' and ratio > .5:
            q.put(("PUSH", 2))

    cap.release()
    q.put(('DONE', None))
    sys.exit()


def imshow(title, frame, pos=None):
    """
    Show the frame with the given title and optional position.
    """
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    """
    Main function to start the factory tool.
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

    # Create a Queue
    q = Queue()
    # Create thread_cam1 and thread_cam2 threads and start them.
    m_thread_cam1 = threading.Thread(target=thread_cam1, args =(q,))
    m_thread_cam2 = threading.Thread(target=thread_cam2, args =(q,))

    m_thread_cam1.start()
    m_thread_cam2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            (name, frame) =q.get()

            # show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            if "VIDEO:" in name:
                imshow(name[6:], frame)

            # Control actuator, name == 'PUSH'
            elif name == "PUSH":
                ctrl.push_actuator(frame)

            elif name == 'DONE':
                FORCE_STOP = True

            q.task_done()

    m_thread_cam1.join()
    m_thread_cam2.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except ImportError:
        os._exit()
