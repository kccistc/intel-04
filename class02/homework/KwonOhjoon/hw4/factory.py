"""
Intel Smart Factory Demo
"""

import os
import threading
from argparse import ArgumentParser
from queue import Queue
from time import sleep

from cv2 import cv2
import numpy as np
import openvino as ov

from iotdemo import FactoryController
from iotdemo import MotionDetector
from iotdemo import ColorDetector


FORCE_STOP = False


def thread_cam1(q):
    """
    Camera 1 thread
    """
    # MotionDetector
    motion = MotionDetector()
    motion.load_preset(path='resources/motion.cfg', key='default')

    # Load and initialize OpenVINO
    # OTX 실습 시간에 생성한 모델을 사용함
    core = ov.Core()
    model = core.read_model('resources/model.xml')
    compiled_model = core.compile_model(model=model, device_name='CPU')
    output_layer = compiled_model.output(0)

    # HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture("resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(('VIDEO:Cam1 live', frame))

        # Motion detect
        detected = motion.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(('VIDEO:Cam1 detected', detected))

        # abnormal detect
        # (1, 3, 224, 224) 형태로 변환
        image = cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(image, (224, 224))
        input_image = np.moveaxis(input_image, -1, 0)
        input_image = np.expand_dims(input_image, axis=0)

        # Inference OpenVINO
        class_name = ['X', 'Circle']
        result_infer = compiled_model([input_image])[output_layer][0]
        result_index = np.argmax(result_infer)

        # Calculate ratios
        print(f"X = {result_infer[0]:.2f}, Circle = {result_infer[1]:.2f}")

        # in queue for moving the actuator 1
        if class_name[result_index] == 'X':
            q.put(('PUSH:1', None))

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    """
    Camera 2 thread
    """
    # MotionDetector
    motion = MotionDetector()
    motion.load_preset(path='resources/motion.cfg', key='default')

    # ColorDetector
    color = ColorDetector()
    color.load_preset(path='resources/color.cfg', key='default')

    # HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture("resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(('VIDEO:Cam2 live', frame))

        # Detect motion
        detected = motion.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(('VIDEO:Cam2 detected', detected))

        # Detect color
        name, ratio = color.detect(detected)[0]
        ratio = ratio * 100

        # Compute ratio
        print(f"{name}: {ratio:.2f}%")

        # Enqueue to handle actuator 2
        if name == 'blue':
            q.put(('PUSH:2', None))

    cap.release()
    q.put(('DONE', None))
    exit()


def imshow(title, frame, pos=None):
    """
    Display frame image
    """
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    """
    Main function
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

    # HW2 Create a Queue
    q = Queue()

    # HW2 Create thread_cam1 and thread_cam2 threads and start them.
    thread1 = threading.Thread(target=thread_cam1, args=(q,))
    thread2 = threading.Thread(target=thread_cam2, args=(q,))

    thread1.start()
    thread2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            # Break the loop if 'q' is pressed.
            if cv2.waitKey(10) & 0xff == ord('q'):
                FORCE_STOP = True
                break

            # dequeue name and data from queue
            name, data = q.get()

            # Show video
            if 'VIDEO' in name:
                imshow(name[6:], data)

            # Control actuator, name == 'PUSH'
            elif 'PUSH' in name:
                num = int(name[5:])
                ctrl.push_actuator(num)
                print(f'Push actuator {num}')   # Terminal output

            elif name == 'DONE':
                FORCE_STOP = True

            q.task_done()

    thread1.join()
    thread2.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
