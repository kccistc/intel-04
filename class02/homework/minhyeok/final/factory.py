'''
2024.06.17 #과제4
'''
#!/usr/bin/env python3

import os
import sys
import threading
from argparse import ArgumentParser
from queue import Queue
from time import sleep

import cv2
import numpy as np
from openvino.runtime import Core
# from openvino.inference_engine import IECore, IENetwork

from iotdemo import FactoryController, MotionDetector, ColorDetector


FORCE_STOP = False


def thread_cam1(q):
    ''' 쓰레드를 사용한 1번 캠입니다. '''
    # MotionDetector
    det = MotionDetector()
    det.load_preset("./resources/motion.cfg", "default")

    # TODO(minhyeok): Load and initialize OpenVINO
    # Load and initialize OpenVINO
    ie = Core()
    model = ie.read_model(model="../homework/minhyeok/hw3/MobileNet-V3-large-1x/openvino.xml",
                          weights="../homework/minhyeok/hw3/MobileNet-V3-large-1x/openvino.bin")
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    output_layer = compiled_model.output(0)

    # Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture('./resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            # cap = cv2.VideoCapture('./resources/conveyor.mp4')
            # _, frame = cap.read()
            break

        # Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live", frame))

        # Motion detect
        detected = det.detect(frame)

        # Enqueue "VIDEO:Cam1 detected", detected info.
        if detected is None:
            continue
        q.put(("VIDEO:Cam1 detected", detected))

        # abnormal detect
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # reshaped = detected[:, :, [2, 1, 0]]
        # np_data = np.moveaxis(reshaped, -1, 0)
        # preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        # batch_tensor = np.stack(preprocessed_numpy, axis=0)

        # TODO(minhyeok): Inference OpenVINO
        if detected is not None:
            # Preprocess the detected frame
            resized_frame = cv2.resize(detected, (224, 224)) # example size, adapt as necessary
            input_tensor = np.expand_dims(resized_frame.transpose(2, 0, 1), axis=0)  # NCHW format

            # Perform inference
            results = compiled_model([input_tensor])[output_layer]

            # Process results (implement your logic here)
            predictions = np.squeeze(results)
            print("Predictions:", predictions)  # or any other processing you need

        # TODO(minhyeok): Calculate ratios
        # print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # TODO(minhyeok): in queue for moving the actuator 1
    cap.release()
    q.put(('DONE', None))
    sys.exit()


def thread_cam2(q):
    ''' 쓰레드를 사용한 2번 캠입니다. '''

    # MotionDetector
    det = MotionDetector()
    det.load_preset("./motion.cfg", "default")

    # ColorDetector
    color = ColorDetector()
    color.load_preset("./color.cfg", "default")
    # Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture('./resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            # cap = cv2.VideoCapture('./resources/conveyor.mp4')
            # _, frame = cap.read()
            break

        # HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live", frame))
        # Detect motion
        detected = det.detect(frame=frame)

        # Enqueue "VIDEO:Cam2 detected", detected info.
        if detected is None:
            continue
        q.put(("VIDEO:Cam2 detected", detected))

        # Detect color
        predict = color.detect(detected)
        name, ratio = predict[0]
        ratio = ratio * 100

        # Compute ratio
        print(f"{name}: {ratio:.2f}%")

        # Enqueue to handle actuator 2
        # 파란색 뚜껑 감지 시 아두이노로 1 전송
        if name == "blue":
            q.put(("PUSH", "1"))
        # 하얀색 뚜껑 감지 시 아두이노로 2 전송
        elif name == "white":
            q.put(("PUSH", "2"))

    cap.release()
    q.put(('DONE', None))
    sys.exit()


def imshow(title, frame, pos=None):
    """ 영상을 화면에 표시 """
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    ''' 메인 코드입니다. '''
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
    t1 = threading.Thread(target=thread_cam1, args=(q, ))
    t2 = threading.Thread(target=thread_cam2, args=(q, ))
    t1.start()
    t2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                FORCE_STOP = True
                break

            # get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            event = q.get()
            name, data = event

            # Show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
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
        os._exit()
