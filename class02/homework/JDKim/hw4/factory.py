#!/usr/bin/env python3

import threading
import sys
from argparse import ArgumentParser
from queue import Queue
from time import sleep

from cv2 import cv2
import numpy as np
import openvino as ov

from iotdemo import ColorDetector, FactoryController, MotionDetector

FORCE_STOP = False


def thread_cam1(q):
    '''
    양품 여부 확인 스레드 
    '''
    # MotionDetector
    det = MotionDetector()
    det.load_preset("./resources/motion.cfg", "default")

    # Load and initialize OpenVINO
    core = ov.Core()
    model = core.read_model(model='./resources/model.xml')

    # Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture('./resources/conveyor.mp4')

    start_flag = True
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

        # Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(("VIDEO:Cam1 detected", detected))

        # Inference OpenVINO
        if start_flag:
            # 'detected' 프레임을 입력 텐서로 변환
            input_tensor = np.expand_dims(detected, 0)  # (H, W, C) -> (1, H, W, C)

            # PrePostProcessor 객체를 생성하여 모델에 대해 전처리 및 후처리 설정
            ppp = ov.preprocess.PrePostProcessor(model)

            # 입력 텐서 설정
            ppp.input().tensor() \
            .set_shape(input_tensor.shape) \
            .set_element_type(ov.Type.u8) \
            .set_layout(ov.Layout('NHWC'))  # noqa: ECE001, N400

            # 입력 데이터 전처리 설정
            ppp.input().preprocess() \
                .resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)  # 선형 보간법을 사용하여 크기 조정

            # 모델 입력 레이아웃 설정
            ppp.input().model().set_layout(ov.Layout('NCHW'))  # 모델 입력 레이아웃 설정 (배치, 채널, 높이, 너비)

            # 출력 텐서 설정
            ppp.output().tensor().set_element_type(ov.Type.f32)  # 출력 데이터 타입 설정 (float32)

            # 전처리 및 후처리 파이프라인을 모델에 적용
            model = ppp.build()  # 모델에 전처리/후처리 정보 적용
            compiled_model = core.compile_model(model=model, device_name="CPU")  # 모델 컴파일 (CPU 타겟)
            start_flag = False  # 모델 준비가 끝났으므로 플래그를 끔

        # 모델 추론 수행
        results = compiled_model.infer_new_request({0: input_tensor})  # 입력 데이터를 사용하여 추론 수행

        # 추론 결과를 추출
        predictions = next(iter(results.values()))  # 결과 딕셔너리에서 첫 번째 값을 가져옴
        probs = predictions.reshape(-1)  # 결과를 1차원 배열로 변환

        # Calculate ratios
        probs *= 100
        print(f'nok: {probs[0]:.2f}, ok: {probs[1]:.2f}')

        # in queue for moving the actuator 1
        if probs[0] > 95:
            q.put(("PUSH", 1))

    cap.release()
    q.put(('DONE', None))
    sys.exit()


def thread_cam2(q):
    '''
    색상 확인 스레드
    '''
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
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(("VIDEO:Cam2 detected", detected))

        # Detect color
        predict = color.detect(detected)
        name, ratio = predict[0]
        ratio = ratio * 100

        # Compute ratio
        print(f"{name}: {ratio:.2f}%")

        # Enqueue to handle actuator 2
        if name == 'blue' and int(ratio) > 50:
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

            try:
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
        sys.exit()
