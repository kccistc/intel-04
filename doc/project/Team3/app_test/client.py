import socket
import cv2
import numpy as np

# 서버 설정
server_address = "서버 IP"
server_port = 서버 Port Number

# 서버 연결
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_address, server_port))

# 응답 받기
response = client_socket.recv(1024).decode("utf-8")
print(f"{response}\n")

# webcam
cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
	try:
		ret, frame = cap.read()
		result, frame = cv2.imencode('.jpg', frame, encode_param)
		data = np.array(frame)
		byteData = data.tobytes()
		
		client_socket.sendall((str(len(byteData))).encode().ljust(16) + byteData)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	except Exception:
		print("[연결 종료]")
		break

cap.release()

# 소켓 닫기
client_socket.close()
