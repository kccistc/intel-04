import pyaudio
import wave
import threading
import time
import cv2
import numpy as np

def record_audio(filename, duration, channels=1, rate=44100, frames_per_buffer=1024):
    p = pyaudio.PyAudio()
    
    def callback(in_data, frame_count, time_info, status):
        frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=frames_per_buffer,
                    stream_callback=callback)
    
    frames = []
    print(f"Recording for {duration} seconds...")
    stream.start_stream()

    # 녹음 지속 시간만큼 대기
    threading.Event().wait(duration)
    
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Recording finished.")

    # WAV 파일로 저장
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Recording saved as {filename}")

def start_audio_recording():
    record_audio("output.wav", duration=5)

recording_thread = None

print('start')

# 이미지 생성
image = np.ones((200, 1200,3), np.uint8) * 255

# 윈도우 생성
cv2.namedWindow("Info")

# 글자 설정
i=5900
text1 = "SYSTEM SHUTDOWN"
text2 = f"Please say SORRY in {i} seconds"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
thickness = 2
text_size, _ = cv2.getTextSize(text2, font, font_scale, thickness)
text_x = (image.shape[1] - text_size[0]) // 2
text_y = (image.shape[0] + text_size[1]) // 3

# 붉은색으로 글자 그리기
color = (0, 0, 255)  # BGR 형식으로 붉은색 지정
cv2.putText(image, text1, (text_x, text_y), font, font_scale, color, thickness)
#cv2.putText(image, f"Please say SORRY in {i} second", (text_x, text_y*2), font, font_scale, color, thickness)
cv2.imshow("Info", image)

recording_thread = threading.Thread(target=start_audio_recording)
recording_thread.start()

# 5초 동안 화면 표시
for _ in range(50):
    cv2.imshow("Info", image)
    i = i-100
    image = np.ones((200, 1200,3), np.uint8) * 255
    cv2.putText(image, text1, (text_x, text_y), font, font_scale, color, thickness)
    cv2.putText(image, f"Please say SORRY in {i//1000} second", (text_x, text_y*2), font, font_scale, color, thickness)
    cv2.waitKey(100)  # 100ms 동안 대기

# 5초 후에 자동으로 종료
#time.sleep(5)
print('end')
cv2.destroyAllWindows()
recording_thread.join()
