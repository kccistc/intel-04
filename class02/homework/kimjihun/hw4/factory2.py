"""
Smart factory
"""

import threading
import sys
from argparse import ArgumentParser
from time import sleep
from queue import Queue

from cv2 import cv2
import numpy as np
from openvino.inference_engine import IECore

from iotdemo import MotionDetector , FactoryController , ColorDetector


FORCE_STOP = False

# thread1 품질 predict
def thread_cam1(q):
    """
    Camera 1 thread.
    """
    # MotionDetector
    det = MotionDetector()
    det.load_preset("./resources/motion.cfg","default")
    # Load and initialize OpenVINO
    ie = IECore()
    model_xml = "resources/model.xml"
    model_bin = "resources/model.bin"
    net = ie.read_network(model=model_xml, weights=model_bin)
    exec_net = ie.load_network(network=net, device_name="CPU")

    input_blob = next(iter(net.input_info))
    output_blob = next(iter(net.outputs))
    # Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture('./resources/conveyor.mp4')
    #thread1 main
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        # no input
        if frame is None:
            break
        # Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live", frame))
        # Motion detect
        #no input detected
        detected = det.detect(frame)
        if detected is None:
            continue
        # Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(("VIDEO: Cam1 detected", detected))
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
            print("1")

    cap.release()
    q.put(('DONE', None))
    sys.exit()

def thread_cam2(q):
    """
    Camera 2 thread.
    """
    # MotionDetector
    det = MotionDetector()
    det.load_preset("./resources/motion.cfg","default")
    # ColorDetector
    color = ColorDetector()
    color.load_preset("./resources/color.cfg","default")
    # Open video clip resources/conveyor.mp4 instead of camera device.
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

        # Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(("VIDEO: Cam2 detected", detected))

        # Detect color
        predict = color.detect(detected)
        name , ratio = predict[0]
        ratio = ratio*100

        # Compute ratio
        print(f"{name}: {ratio:.2f}%")

        # Enqueue to handle actuator 2
        if name == 'blue':
            q.put(("PUSH", 2))
            print("2")
    cap.release()
    q.put(('DONE', None))
    sys.exit()

# prin image
def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)

# main code
def main():
    """
    main code
    """
    global FORCE_STOP

    # set arg
    parser = ArgumentParser(prog='python3 factory.py', description="Factory tool")
    # set arg
    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()

    # Create a Queue
    q = Queue()

    # Create thread_cam1 and thread_cam2 threads and start them.
    t1 = threading.Thread(target=thread_cam1,args=(q, ))
    t2 = threading.Thread(target=thread_cam2,args=(q, ))
    #thread start
    t1.start()
    t2.start()

    # fac controll
    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                FORCE_STOP = True
                break

            # get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            event = q.get()
            name, data = event

             # show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            if "VIDEO:" in name:
                imshow(name[6:], data)

            # Control actuator, name == 'PUSH'
            elif name == "PUSH":
                ctrl.push_actuator(data)

            elif name == 'DONE':
                FORCE_STOP = True

            # done
            q.task_done()

    # init thread
    t1.join()
    t2.join()
    cv2.destroyAllWindows()

# main loop
if __name__ == '__main__':
    try:
        main()
    except Exception:
        sys.exit()
