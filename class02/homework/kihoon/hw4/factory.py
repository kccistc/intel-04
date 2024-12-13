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
    det = MotionDetector()
    det.load_preset("motion.cfg", "default")

    ie = IECore()
    model = ie.read_network(model="resources/model.xml", weights="resources/model.bin")
    exec_net = ie.load_network(network=model, device_name="CPU")
    input_blob = next(iter(model.input_info))
    output_blob = next(iter(model.outputs))
    cap = cv2.VideoCapture('resources/conveyor.mp4')
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

        # Enqueue "VIDEO:Cam1 detected", detected info
        q.put(("VIDEO:Cam1 detected", detected))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        reshaped = detected[:, :, [2, 1, 0]]
        np_data = np.moveaxis(reshaped, -1, 0)
        preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        batch_tensor = np.stack(preprocessed_numpy, axis=0)
        # Inference OpenVINO

        results = exec_net.infer(inputs={input_blob: batch_tensor})
        output_data = results[output_blob]
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
    # MotionDetector
    det = MotionDetector()
    det.load_preset("motion.cfg", "default")

    # ColorDetector
    color = ColorDetector()
    color.load_preset("color.cfg", "default")
    cap = cv2.VideoCapture('./resources/conveyor.mp4')

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

        # Enqueue "VIDEO:Cam2 detected", detected info
        q.put(("VIDEO:Cam2 detected", detected))

        # Detect color
        predict = color.detect(detected)

        # Compute ratio
        name, ratio = predict[0]
        ratio = ratio * 100
        print(f"{name}: {ratio:.2f}%")

        # Enqueue to handle actuator 2
        if name == "White" and ratio < 0.5:
            q.put(("PUSH", 2))

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

    # Create a Queue
    q = Queue()
    # Create thread_cam1 and thread_cam2 threads and start them
    m_thread_cam1 = threading.Thread(target=thread_cam1, args=(q,))
    m_thread_cam2 = threading.Thread(target=thread_cam2, args=(q,))
    m_thread_cam1.start()
    m_thread_cam2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            try:
                name, data = q.get(timeout=1)
            except Empty:
                continue

            if name.startswith("VIDEO:"):
                imshow(name[6:], data, (0, 0) if name.endswith("Cam1 live") else (640, 0))
            elif name == "PUSH":
                if data == 1:
                    ctrl.push_actuator_1()
                elif data == 2:
                    ctrl.push_actuator_2()
            elif name == 'DONE':
                FORCE_STOP = True
                q.task_done()

    m_thread_cam1.join()
    m_thread_cam2.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        os._exit(1)