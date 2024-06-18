'''
Smart Factory HW Final
'''
#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
from openvino.runtime import Core,Type,Layout
from openvino.preprocess import PrePostProcessor,ResizeAlgorithm

from iotdemo import FactoryController
from iotdemo.motion import MotionDetector
from iotdemo.color.color_detector import ColorDetector

FORCE_STOP = False

def thread_cam1(q):
    """Thread function for Camera 1"""
    md = MotionDetector()
    md.load_preset()

    core = Core()

    model_xml = "resources/model.xml"
    model_bin = "resources/model.bin"

    model = core.read_model(model_xml,model_bin)

    ppp = PrePostProcessor(model)

    ppp.input().tensor() \
        .set_element_type(Type.u8) \
        .set_layout(Layout('NHWC'))

    ppp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)

    ppp.input().model().set_layout(Layout('NCHW'))

    ppp.output().tensor().set_element_type(Type.f32)

    model = ppp.build()

    compiled_model = core.compile_model(model, 'CPU')

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

        input_tensor = np.expand_dims(detected, 0)

        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)*100

        print(f"X = {probs[0]:.2f}%, Circle = {probs[1]:.2f}%")

        if probs[0] > probs[1]:
            q.put(("PUSH", 1))

    cap.release()
    q.put(('DONE', None))
    os._exit(0)


def thread_cam2(q):
    """Thread function for Camera 2"""
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
    os._exit(0)


def imshow(title, frame, pos=None):
    """Helper function to show image frames"""
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    """Main function to run the factory tool"""
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
                name, data = q.get_nowait()

                if "VIDEO:" in name:
                    imshow(name[6:], data)

                if name == 'PUSH':
                    ctrl.push_actuator(data)

                if name == 'DONE':
                    FORCE_STOP = True

                q.task_done()

            except Empty:
                continue

    t1.join()
    t2.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        os._exit(1)
