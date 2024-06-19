#!/usr/bin/env python3

import sys
import threading
from argparse import ArgumentParser
from queue import Queue
from time import sleep

from cv2 import cv2 
import numpy as np
from openvino.inference_engine import IECore

from iotdemo import FactoryController
from iotdemo.motion import MotionDetector
from iotdemo.color.color_detector import ColorDetector

FORCE_STOP = False


def thread_cam1(q):
    md = MotionDetector()
    md.load_preset()

    ie = IECore()
    model_xml = "resources/model.xml"
    model_bin = "resources/model.bin"

    net = ie.read_network(model=model_xml, weights=model_bin)
    exec_net = ie.load_network(network=net, device_name="CPU")
    input_blob = next(iter(net.input_info))
    output_blob = next(iter(net.outputs))

    cap = cv2.VideoCapture('resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q.put(("VIDEO:Cam1 live", frame))

        detected = md.detect(frame)
        if detected is None:
            continue

        q.put(("VIDEO:Cam1 detected", detected))

        reshaped = detected[:, :, [2, 1, 0]]
        np_data = np.moveaxis(reshaped, -1, 0)
        preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        batch_tensor = np.stack(preprocessed_numpy, axis=0)

        result = exec_net.infer(inputs={input_blob: batch_tensor})
        output_data = result[output_blob]
        x_ratio = output_data[0][0]
        circle_ratio = output_data[0][1]

        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        if x_ratio > circle_ratio:
            q.put(("PUSH", 1))

    cap.release()
    q.put(('DONE', None))
    sys.exit()


def thread_cam2(q):
    md = MotionDetector()
    md.load_preset()

    cd = ColorDetector()
    cd.load_preset()

    cap = cv2.VideoCapture('resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        q.put(("VIDEO:Cam2 live", frame))

        detected = md.detect(frame)
        if detected is None:
            continue

        q.put(("VIDEO:Cam2 detected", detected))

        predict = cd.detect(detected)
        name, ratio = predict[0]
        ratio = ratio * 100

        print(f"{name}: {ratio:.2f}%")

        if name == 'blue':
            q.put(("PUSH", 2))

    cap.release()
    q.put(('DONE', None))
    sys.exit()


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

    q = Queue()

    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))

    t1.start()
    t2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            try:
                name, data = q.get(timeout=1)
            except Queue.Empty:
                continue

            if "VIDEO:" in name:
                imshow(name[6:], data)

            if name == 'PUSH':
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
    except Exception as e:
        print(f"Exception occurred: {e}")
        sys.exit(1)
