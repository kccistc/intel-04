import sys
import threading
from argparse import ArgumentParser
from queue import Queue, Empty
from time import sleep

import cv2
import numpy as np
import openvino as ov

from iotdemo import ColorDetector, FactoryController, MotionDetector

FORCE_STOP = False


def thread_cam1(q):
    """
    Thread function for processing camera 1 input.
    Detects motion and performs inference using OpenVINO model.
    """
    # MotionDetector
    det = MotionDetector()
    det.load_preset("./resources/motion.cfg", "default")

    # Load and initialize OpenVINO model
    core = ov.Core()
    model_path = "./resources/model.xml"
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, "CPU")

    # Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture('./resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        reshaped = detected[:, :, [2, 1, 0]]
        np_data = np.moveaxis(reshaped, -1, 0)
        preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        batch_tensor = np.stack(preprocessed_numpy, axis=0)

        # Inference using OpenVINO
        results = compiled_model.infer_new_request({0: batch_tensor})
        predictions = next(iter(results.values())).reshape(-1)
        x_ratio = predictions[0]
        circle_ratio = predictions[1]

        # Calculate ratios
        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # Push actuator 1 if certain conditions are met
        if x_ratio >= circle_ratio:
            q.put(("PUSH", 1))

    cap.release()
    q.put(('DONE', None))
    sys.exit()


def thread_cam2(q):
    """
    Thread function for processing camera 2 input.
    Detects motion and color.
    """
    # MotionDetector
    det = MotionDetector()
    det.load_preset("./resources/motion.cfg", "default")

    # ColorDetector
    color = ColorDetector()
    color.load_preset("./resources/color.cfg", "default")

    # Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture('./resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # Enqueue "VIDEO:Cam2 live", frame info
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
        if name == "blue":
            q.put(("PUSH", 2))

    cap.release()
    q.put(('DONE', None))
    sys.exit()


def imshow(title, frame, pos=None):
    """
    Show the frame with the given title and optional position.
    """
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    """
    Main function to start the factory tool.
    """
    global FORCE_STOP

    parser = ArgumentParser(
        prog='python3 factory.py',
        description="Factory tool"
    )

    parser.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="Arduino port"
    )

    args = parser.parse_args()

    # Create a Queue
    q = Queue()

    # Create thread_cam1 and thread_cam2 threads and start them.
    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))

    t1.start()
    t2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            try:
                # get an item from the queue.
                event = q.get(timeout=0.1)
            except Empty:
                continue

            name, data = event

            # Show videos with titles of 'Cam1 live'
            #           and 'Cam2 live' respectively.
            if "VIDEO:" in name:
                imshow(name[6:], data)
            elif "PUSH" in name:
                ctrl.push_actuator(data)
                print(name)
            elif name == 'DONE':
                FORCE_STOP = True

            q.task_done()

    t1.join()
    t2.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
