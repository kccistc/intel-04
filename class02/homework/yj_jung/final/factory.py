#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
from openvino.inference_engine import IECore
import openvino

from iotdemo import FactoryController
from iotdemo.motion import MotionDetector
from iotdemo.color.color_detector import ColorDetector

FORCE_STOP = False


def thread_cam1(q):
    # TODO: MotionDetector
    md = MotionDetector()
    md.load_preset()
    # TODO: Load and initialize OpenVINO

    ie = IECore()

    model_xml = "resources/model.xml"
    model_bin = "resources/model.bin"

    net = ie.read_network(model=model_xml, weights=model_bin)

    exec_net = ie.load_network(network=net, device_name="CPU")

    # 입력 레이어 이름 추출
    input_blob = next(iter(net.input_info))

    # 출력 레이어 이름 추출
    output_blob = next(iter(net.outputs))
    
    

    cap = cv2.VideoCapture('resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        q.put(("VIDEO:Cam1 live",frame))

        # TODO: Motion detect
        detected = md.detect(frame)
        if detected is None:
            continue

        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(("VIDEO:Cam1 detected",detected))
    
        reshaped = detected[:, :, [2, 1, 0]]

        np_data = np.moveaxis(reshaped, -1, 0)
        preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        batch_tensor = np.stack(preprocessed_numpy, axis=0)

        # TODO: Inference OpenVINO
        result = exec_net.infer(inputs={input_blob: batch_tensor})
        output_data = result[output_blob]
        x_ratio = output_data[0][0]
        circle_ratio = output_data[0][1]

        # TODO: Calculate ratios
        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # TODO: in queue for moving the actuator 1
        if x_ratio > circle_ratio:
            q.put(("PUSH",1))

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    # TODO: MotionDetector
    md = MotionDetector()
    md.load_preset()

    # TODO: ColorDetector
    cd = ColorDetector()
    cd.load_preset()

    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture('resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live",frame))

        # TODO: Detect motion
        detected = md.detect(frame)

        if detected is None:
            continue

        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(("VIDEO:Cam2 detected",detected))

        # TODO: Detect color
        predict = cd.detect(detected)
        name,ratio = predict[0]
        ratio = ratio * 100

        # TODO: Compute ratio
        print(f"{name}: {ratio:.2f}%")

        # TODO: Enqueue to handle actuator 2
        if name == 'blue':
            q.put(("PUSH",2))

    cap.release()
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
    t1 = threading.Thread(target=thread_cam1,args=(q,))
    t2 = threading.Thread(target=thread_cam2,args=(q,))

    t1.start()
    t2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # TODO: HW2 get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            name,data = q.get()

            # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            if "VIDEO:" in name:
                imshow(name[6:],data)

            # TODO: Control actuator, name == 'PUSH'

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
    except Exception:
        os._exit()
