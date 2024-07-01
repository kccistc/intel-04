import cv2 # type: ignore
import json
import time
from ultralytics import YOLO # type: ignore
import logging
import os
import socket

server_address = 'localhost'
server_port = 8888
buffer_size = 1024

def socket_com(json_data):
    global server_address
    global server_port
    global buffer_size

    # 소켓 생성
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            # 서버에 연결
            client_socket.connect((server_address, server_port))
            print(f"Connected to server at {server_address}:{server_port}")

            # JSON 데이터 전송
            client_socket.sendall(json_data.encode())
            print(f"Sent to server: {json_data}")
            
            #time.sleep(10)
            
            client_socket.close()


        except Exception as e:
            print(f"An error occurred: {e}")
            
def edit_json(message_type, value):
    data = {
        "message_type": message_type,
        "value": value
    }
    json_data = json.dumps(data)
    return json_data

# 로깅 설정
logging.basicConfig(filename='video_processing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
# 클래스 이름 정의
class_names = ['guide dog', 'whiteCane', 'wheelchair']
model = YOLO('./best.pt')  # YOLO 모델 경로 설정

# 웹캠 설정
cap = cv2.VideoCapture(0)  # 웹캠 디바이스 ID 설정, 기본적으로 0

if not cap.isOpened():  # 웹캠 열기 실패 시
    logger.error("웹캠을 열 수 없습니다.")  # 오류 로깅
    exit()  # 프로그램 종료

# 객체 탐지 결과를 저장할 리스트
detections = []

# 탐지 주기 설정 (프레임당 대기 시간)
detection_interval = 3.0  # 초단위 설정 (예: 1초)

last_detection_time = time.time()  # 마지막 탐지 시간 초기화

while True:
    ret, frame = cap.read()  # 프레임 읽기

    if not ret:  # 프레임 읽기 실패 시
        logger.error("프레임을 가져올 수 없습니다.")  # 오류 로깅
        break  # 루프 종료

    # 현재 시간
    current_time = time.time()

    # 프레임당 대기 시간이 지나면 객체 탐지 실행
    if current_time - last_detection_time >= detection_interval:
        # 모델 예측
        results = model.predict(frame,verbose=False)

        for result in results:
            for box in result.boxes:
                if box.conf > 0.7:  # 신뢰도가 0.7 이상인 경우
                    class_id = int(box.cls[0])
                    class_name = class_names[class_id]

                    # 객체 탐지 결과를 리스트에 추가
                    detections.append((class_name, box.conf, box.xyxy[0]))

        # 마지막 탐지 시간 업데이트
        last_detection_time = current_time

    # 지정된 시간 동안의 객체 탐지 결과 평가 및 메시지 전송
    if detections:
        # 최근 탐지된 객체 중에서 가장 높은 신뢰도의 객체 선택
        best_detection = max(detections, key=lambda x: x[1])

        class_name, conf, bbox = best_detection
        x_min, y_min, x_max, y_max = map(int, bbox)

        # 객체 탐지에 따른 JSON 데이터 생성 및 전송
        if class_name == 'wheelchair':
            print("wheel_chair_message create!!!")
            json_data = edit_json("client_classification",'Wheelchair')
            socket_com(json_data)
            cap.release()  # 웹캠 해제
            cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
            break

        elif class_name in ['whiteCane', 'guide dog']:
            json_data = edit_json("client_classification",'Blind')
            socket_com(json_data)
            cap.release()  # 웹캠 해제
            cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
            break

        # 탐지 결과 초기화
        detections = []

    # 결과를 포함한 프레임 디스플레이
    cv2.imshow('YOLO Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 루프 종료
        break

cap.release()  # 웹캠 해제
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
