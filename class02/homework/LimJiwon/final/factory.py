"""
Factory module

This module contains functions and classes related to factory operations,
including video processing and device control.
"""

#!/usr/bin/env python3

import sys
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
import openvino as ov

from iotdemo import FactoryController, MotionDetector, ColorDetector

FORCE_STOP = False


def thread_cam1(q):
    """
    Process the input image by detecting good or defective.
    """

    # MotionDetector
    det = MotionDetector()
    det.load_preset("./resources/motion.cfg", "default")

    # Load and initialize OpenVINO

    core = ov.Core()
    model = core.read_model("./resources/model.xml")

    # Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture("./resources/conveyor.mp4")

    start_flag = True
    compiled_model = None

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
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # reshaped = detected[:, :, [2, 1, 0]]
        # np_data = np.moveaxis(reshaped, -1, 0)
        # preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        # batch_tensor = np.stack(preprocessed_numpy, axis=0)

        # Inference OpenVINO
        if start_flag:
            input_tensor = np.expand_dims(detected, 0)
            ppp = ov.preprocess.PrePostProcessor(model)

            ppp.input().tensor() \
                .set_shape(input_tensor.shape) \
                .set_element_type(ov.Type.u8) \
                .set_layout(ov.Layout('NHWC'))  # noqa: ECE001, N400

            ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
            ppp.input().model().set_layout(ov.Layout('NCHW'))
            ppp.output().tensor().set_element_type(ov.Type.f32)

            model = ppp.build()
            compiled_model = core.compile_model(model, "CPU")
            start_flag = False

        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)

        # Calculate ratios
        # x_ratio : probs[0] * 100, circle_ratio : probs[1] * 100
        print(f"X = {probs[0] * 100:.2f}%, Circle = {probs[1] * 100:.2f}%")

        # in queue for moving the actuator 1
        if 0 < probs[0]:
            q.put(("PUSH", 1))


    cap.release()
    q.put(('DONE', None))
    sys.exit(0)


def thread_cam2(q):
    """
    Process the input image by detecting blue or white.
    """

    # MotionDetector
    det = MotionDetector()
    det.load_preset("./resources/motion.cfg", "default")

    # ColorDetector
    color = ColorDetector()
    color.load_preset("./resources/color.cfg", "default")

    # Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture("./resources/conveyor.mp4")

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

        # TDetect color
        predict = color.detect(detected)

        # Compute ratio
        name, ratio = predict[0]
        ratio = ratio * 100
        print(f"{name}: {ratio:.2f}%")

        # Enqueue to handle actuator 2
        if name == "blue":
            q.put(("PUSH", 2))

    cap.release()
    q.put(('DONE', None))
    sys.exit(0)


def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    """
    main
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

    #Create a Queue
    q = Queue()

    # Create thread_cam1 and thread_cam2 threads and start them.
    t1 = threading.Thread(target=thread_cam1, args=(q, ))
    t2 = threading.Thread(target=thread_cam2, args=(q, ))

    t1.start()
    t2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            try:
                event = q.get_nowait()
            except Empty:
                continue

            name, data = event

            # show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            if "VIDEO:" in name:
                imshow(name[6:], data)

            # Control actuator, name == 'PUSH'
            elif name == "PUSH":
                ctrl.push_actuator(data)

            if name == 'DONE':
                FORCE_STOP = True

            q.task_done()

    t1.join()
    t2.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        sys.exit(1)
