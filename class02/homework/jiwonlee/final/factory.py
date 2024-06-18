#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Queue
from time import sleep
import sys
from cv2 import cv2
import numpy as np
import openvino as ov
from iotdemo import FactoryController, MotionDetector, ColorDetector

FORCE_STOP = False

def thread_cam1(queue):
    """
    Thread function for camera 1.
    Uses MotionDetector to detect motion in the video.
    """
    det = MotionDetector()
    det.load_preset("resources/motion.cfg", "default")

    # Load and initialize OpenVINO
    core = ov.Core()
    model = core.read_model(model="resources/model.xml")

    # Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture('./resources/conveyor.mp4')

    start_flag = True
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # Enqueue "VIDEO:Cam1 live", frame info
        queue.put(("VIDEO:Cam1 live", frame))

        # Motion detect
        detected = det.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam1 detected", detected info.
        queue.put(("VIDEO:Cam1 detected", detected))
        # abnormal detect
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        reshaped = detected[:, :, [2, 1, 0]]
        np_data = np.moveaxis(reshaped, -1, 0)
        preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        batch_tensor = np.stack(preprocessed_numpy, axis=0)

        # Inference OpenVINO
        input_tensor = np.expand_dims(detected, 0)

        if start_flag is True:
            ppp = ov.preprocess.PrePostProcessor(model)

            ppp.input().tensor() \
                .set_shape(input_tensor.shape) \
                .set_element_type(ov.Type.u8) \
                .set_layout(ov.Layout('NHWC'))  # noqa: ECE001, N400
            ppp.input().preprocess()\
                .resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
            ppp.input().model().set_layout(ov.Layout('NCHW'))
            ppp.output().tensor().set_element_type(ov.Type.f32)

            model = ppp.build()
            compiled_model = core.compile_model(model, "CPU")
            start_flag = False

        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)
        probs *= 100

        # Calculate ratios
        #print(f"X = {probs[0]:.2f}, Circle: {probs[1]:.2f}")

        # in queue for moving the actuator 1
        if probs[0] >= 50:
            queue.put(("PUSH", 1))
            print("Bad", probs[0])
        else:
            print("Good", probs[1])

    cap.release()
    queue.put(('DONE', None))
    sys.exit()


def thread_cam2(queue):
    """
    Thread function for camera 2.
    Uses MotionDetector and ColorDetector to detect motion and color in the video.
    """
    det = MotionDetector()
    det.load_preset("resources/motion.cfg", "default")

    # ColorDetector
    color = ColorDetector()
    color.load_preset("resources/color.cfg", "default")

    # Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture("resources/conveyor.mp4")
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # Enqueue "VIDEO:Cam2 live", frame info
        queue.put(("VIDEO:Cam2 live", frame))
        # Detect motion
        detected = det.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam2 detected", detected info.
        queue.put(("VIDEO:Cam2 detected", detected))

        # Detect color
        predict = color.detect(detected)
        name, ratio = predict[0]
        ratio = ratio * 100

        # Compute ratio
        print(f"{name}: {ratio:.2f}%")

        # Enqueue to handle actuator 2
        if name == 'blue' and ratio > .5:
            queue.put(("PUSH", 2))

    cap.release()
    queue.put(('DONE', None))
    sys.exit()


def imshow(title, frame, pos=None):
    """
    Display the image in a window.
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

    # Create a Queue
    queue = Queue()

    # Create thread_cam1 and thread_cam2 threads and start them.
    t1 = threading.Thread(target=thread_cam1, args=(queue,))
    t2 = threading.Thread(target=thread_cam2, args=(queue,))
    t1.start()
    t2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            event = queue.get()
            name, data = event

            # show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            if "VIDEO:" in name:
                imshow(name[6:], data)

            # Control actuator, name == 'PUSH'
            elif name == "PUSH":
                ctrl.push_actuator(data)
            elif name == 'DONE':
                FORCE_STOP = True

            queue.task_done()

    t1.join()
    t2.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit(1)
