#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Queue
from time import sleep
import cv2
import numpy as np
# from openvino.inference_engine import IECore

from iotdemo import FactoryController, MotionDetector, ColorDetector

FORCE_STOP = False

def thread_cam1(q):
    det = MotionDetector()
    det.load_preset('motion.cfg', 'default')

    cap = cv2.VideoCapture('./resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break
        q.put(("VIDEO:Cam1 live", frame))
        detected = det.detect(frame)
        if detected is None:
            continue
        q.put(("VIDEO:Cam1 detected", detected))
        # Further processing can be added here

    cap.release()
    q.put(('DONE', None))

def thread_cam2(q):
    det = MotionDetector()
    det.load_preset('motion.cfg', 'default')
    color = ColorDetector()
    color.load_preset('./resources/color.cfg', 'default')
    
    cap = cv2.VideoCapture('./resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break
        q.put(("VIDEO:Cam2 live", frame))
        detected = det.detect(frame)
        if detected is None:
            continue
        q.put(("VIDEO:Cam2 detected", detected))
        color_result = color.detect(detected)
        if color_result:
            name, ratio = color_result[0]
            if name == "blue":
                q.put(("PUSH", 2))
        else:
            print("No color detected")
    
    cap.release()
    q.put(('DONE', None))

def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)

def main():
    global FORCE_STOP

    parser = ArgumentParser(prog='python3 factory.py', description="Factory tool")
    parser.add_argument("-d", "--device", default=None, type=str, help="Arduino port")
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
                event = q.get(timeout=1)
                name, data = event
                if "VIDEO" in name:
                    imshow(name[6:], data)
                elif name == "PUSH":
                    ctrl.push_actuator(data)
                if name == 'DONE':
                    FORCE_STOP = True
                q.task_done()
            except Exception as e:
                print(f"Error: {e}")

    t1.join()
    t2.join()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        os._exit(1)
