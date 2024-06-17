'''
Smart Factory HW4
'''
#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep
import sys

from cv2 import cv2
import numpy as np
from openvino.inference_engine import IECore

from iotdemo import ColorDetector, FactoryController, MotionDetector

FORCE_STOP = False

def thread_cam1(q):
    '''
    Thread for cam1
    '''
    # MotionDetector
    det = MotionDetector()
    det.load_preset(path='./resources/motion.cfg',key = "default")
    # Load and initialize OpenVINO
    ie = IECore()
    path_to_xml_file = './resources/model.xml'
    path_to_bin_file = './resources/model.bin'
    net = ie.read_network(model=path_to_xml_file, weights=path_to_bin_file)
    exec_net = ie.load_network(network=net, device_name="CPU", num_requests=0)
    input_blob = next(iter(net.input_info))
    output_blob = next(iter(net.outputs))
    # HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture("./resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # HW2 Enqueue "VIDEO:Cam1 live", frame info
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
        result = exec_net.infer(inputs={input_blob: batch_tensor})
        predictions = result[output_blob]
        x_ratio, circle_ratio = predictions[0]

        # Calculate ratios
        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # in queue for moving the actuator 1
        q.put(("PUSH", 1))

    cap.release()
    q.put(('DONE', None))
    sys.exit()


def thread_cam2(q):
    '''
    Thread for cam2
    '''
    # MotionDetector
    det = MotionDetector()
    det.load_preset(path='./resources/motion.cfg',key = "default")

    # ColorDetector
    color = ColorDetector()
    color.load_preset(path='./resources/color.cfg', key='default')

    # HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture("./resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live", frame))

        # Detect motion
        detected = det.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(("VIDEO:Cam2 detected", detected))

        # Detect color
        predict= color.detect(detected)
        name, ratio = predict[0]

        # Compute ratio
        ratio = ratio * 100
        print(f"{name}: {ratio:.2f}%")

        # Enqueue to handle actuator 2
        q.put(("PUSH", 2))

    cap.release()
    q.put(('DONE', None))
    sys.exit()


def imshow(title, frame, pos=None):
    '''
    Image Show
    '''
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    '''
    Main Function
    '''
    global FORCE_STOP

    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()

    # HW2 Create a Queue
    q = Queue()

    # HW2 Create thread_cam1 and thread_cam2 threads and start them.
    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # HW2 get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            try :
                event = q.get()
            except Empty:
                continue
            name, data = event

            # HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            if "VIDEO:" in name:
                imshow(name[6:], data)
            # Control actuator, name == 'PUSH'
            elif name == "PUSH" :
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
    except ImportError:
        os._exit()
