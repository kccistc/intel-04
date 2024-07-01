import os  # 운영체제와 상호작용하기 위한 모듈
import json  # JSON 데이터를 다루기 위한 모듈
import pexpect  # 다른 프로그램을 실행하고 제어하기 위한 모듈
import speech_recognition as sr  # 음성 인식을 위한 모듈
from gtts import gTTS  # Google Text-to-Speech를 위한 모듈
import playsound  # 사운드 재생을 위한 모듈
from watchdog.observers import Observer  # 파일 시스템 이벤트를 감시하기 위한 모듈
from watchdog.events import FileSystemEventHandler  # 파일 시스템 이벤트 핸들러를 정의하기 위한 모듈
import subprocess  # 서브 프로세스를 생성하고 관리하기 위한 모듈
import threading  # 멀티스레딩을 위한 모듈

# ----- modified ----- #
import socket
# ----- modified ----- #

class VoiceToTextToVoice:
    def __init__(self, local_input_json, remote_server, remote_path, remote_user, remote_password):
        self.local_input_json = local_input_json  # 로컬 입력 JSON 파일 경로
        self.remote_server = remote_server  # 원격 서버 주소
        self.remote_path = remote_path  # 원격 서버의 경로
        self.remote_user = remote_user  # 원격 서버의 사용자명
        self.remote_password = remote_password  # 원격 서버의 비밀번호

        # ----- modified ----- #
        self.voice_start_mp3 = os.path.join(os.getcwd(), "voice_wait.mp3")  # 시작 음성 파일 경로
        self.response_mp3 = os.path.join(os.getcwd(), "response.mp3")  # 응답 음성 파일 경로
        self.voice_wait = os.path.join(os.getcwd(), "voice_wait.mp3")  # 음성 서비스 대기 파일 경로
        # ----- modified ----- #
        self.audio_playing = False  # 오디오 재생 상태를 추적하는 플래그

    def play_sound(self, file_path):
        self.audio_playing = True  # 오디오 재생 중임을 나타내는 플래그 설정
        # 서브 프로세스를 사용하여 playsound 실행 및 stderr를 DEVNULL로 리디렉션하여 ALSA 경고 메시지를 무시
        subprocess.run(['python3', '-m', 'playsound', file_path], stderr=subprocess.DEVNULL)
        self.audio_playing = False  # 오디오 재생 완료 후 플래그 해제

    def record_voice(self):
        r = sr.Recognizer()  # 음성 인식기 객체 생성
        with sr.Microphone() as source:  # 마이크로부터 음성을 입력받기 위한 컨텍스트 매니저
            if os.path.exists(self.voice_start_mp3):  # 시작 음성 파일이 존재하는지 확인
                self.play_sound(self.voice_start_mp3)  # 시작 음성 파일 재생
            else:
                print(f"{self.voice_start_mp3} 파일이 존재하지 않습니다.")  # 파일이 없을 경우 메시지 출력
            print("마이크에서 음성을 입력할 준비를 하세요...")  # 음성 입력 준비 메시지 출력
            
            print("지금부터 음성을 입력하세요.")  # 음성 입력 시작 메시지 출력
            audio = r.listen(source,timeout=10)  # 음성 입력을 듣기 시작
            print("음성 입력이 완료되었습니다.")  # 음성 입력 완료 메시지 출력

        try:
            question = r.recognize_google(audio, language='ko-KR')  # 구글 음성 인식 API를 사용하여 음성을 텍스트로 변환
            print(f"인식된 질문: {question}")  # 인식된 질문 출력
            return question  # 인식된 질문 반환
        except sr.UnknownValueError:  # 음성을 인식할 수 없는 경우
            print("음성을 인식할 수 없습니다.")  # 오류 메시지 출력
            return ""
        except sr.RequestError as e:  # 음성 인식 API 요청 실패 시
            print(f"구글 음성 인식 API 요청에 실패했습니다: {e}")  # 오류 메시지 출력
            return ""

    def save_input_json(self, question):
        input_data = {
            "questions": [
                {
                    "Q": question  # 인식된 질문을 JSON 형식으로 저장
                }
            ]
        }
        with open(self.local_input_json, "w", encoding="utf-8") as f:  # JSON 파일을 쓰기 모드로 열기
            json.dump(input_data, f, ensure_ascii=False, indent=4)  # JSON 데이터를 파일에 저장
        print(f"{self.local_input_json} 파일에 질문이 저장되었습니다.")  # 저장 완료 메시지 출력

    def send_input_json(self):
        command = f"scp {self.local_input_json} {self.remote_user}@{self.remote_server}:{self.remote_path}"  # 파일 전송 명령어 생성
        child = pexpect.spawn(command)  # pexpect를 사용하여 명령어 실행
        child.expect("password:")  # 비밀번호 입력 프롬프트 대기
        child.sendline(self.remote_password)  # 비밀번호 입력
        child.wait()  # 명령어 실행 완료 대기
        print("input.json 파일을 서버로 전송했습니다.")  # 전송 완료 메시지 출력

    def speak_response(self, text):
        tts = gTTS(text=text, lang='ko', slow=True)  # 텍스트를 음성으로 변환 (천천히 출력)
        tts.save(self.response_mp3)  # 변환된 음성을 파일로 저장
        print("응답을 음성으로 재생합니다.")  # 응답 음성 재생 메시지 출력
        self.play_sound(self.response_mp3)  # 응답 음성 파일 재생
#        os.remove(self.response_mp3)  # 응답 음성 파일 삭제

class OutputFileHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback  # 콜백 함수를 저장

    def on_modified(self, event):
        if event.src_path.endswith("output.json"):  # 변경된 파일이 output.json인지 확인
            print("output.json 파일이 변경되었습니다.")  # 변경 메시지 출력
            self.callback()  # 콜백 함수 호출

class VoiceAssistant:
    def __init__(self, input_json, output_json, remote_server, remote_path, remote_user, remote_password):
        self.input_json = os.path.join(os.getcwd(), input_json)  # 입력 JSON 파일 경로
        self.output_json = os.path.join(os.getcwd(), output_json)  # 출력 JSON 파일 경로
        self.remote_server = remote_server  # 원격 서버 주소
        self.remote_path = remote_path  # 원격 서버 경로
        self.remote_user = remote_user  # 원격 서버 사용자명
        self.remote_password = remote_password  # 원격 서버 비밀번호
        self.voice_to_text_to_voice = VoiceToTextToVoice(input_json, remote_server, remote_path, remote_user, remote_password)  # VoiceToTextToVoice 객체 생성
        self.question = ""
        self.answer = ""

        # ----- modified ----- #
        self.server_address = '127.0.0.1'
        self.server_port = 8888
        self.buffer_size = 1024
        # ----- modified ----- #
        
        self.lock = threading.Lock()  # 락 객체 생성

        self.setup_observer()  # 파일 시스템 감시자 설정

    def setup_observer(self):
        event_handler = OutputFileHandler(self.process_output_json)  # OutputFileHandler 객체 생성
        observer = Observer()  # 파일 시스템 감시자 객체 생성
        observer.schedule(event_handler, path=os.path.abspath('.'), recursive=False)  # 절대 경로로 감시할 경로 설정
        observer.start()  # 감시자 시작
        print("파일 시스템 모니터링을 시작합니다.")  # 감시자 시작 메시지 출력
        self.observer = observer  # 감시자 객체 저장

    def edit_json(self,message_type, question, answer):
        data = {
            "message_type": message_type,
            "question": question,
            "answer": answer
        }
        json_data = json.dumps(data)
        return json_data

    def socket_com(self, json_data):
        # 소켓 생성
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            try:
                # 서버에 연결
                client_socket.connect((self.server_address, self.server_port))
                print(f"Connected to server at {self.server_address}:{self.server_port}")

                # JSON 데이터 전송
                client_socket.sendall(json_data.encode())
                print(f"Sent to server: {json_data}")
                
                client_socket.close()

            except Exception as e:
                print(f"An error occurred: {e}")

    def process_output_json(self):
        with self.lock:
            with open(self.output_json, "r", encoding="utf-8") as f:  # 출력 JSON 파일을 읽기 모드로 열기
                output_data = json.load(f)  # JSON 데이터를 로드
            if isinstance(output_data, list) and len(output_data) > 0 and "answer" in output_data[0]:  # 데이터가 리스트 형식이고 적어도 하나의 항목이 있으며 "answer" 키가 존재하는지 확인
                answer = output_data[0]['answer']  # 응답 텍스트 추출
                self.answer = answer
                json_data = self.edit_json("voice_qna",self.question,self.answer)
                self.socket_com(json_data)
                print(f"응답: {answer}")  # 응답 텍스트 출력
                self.voice_to_text_to_voice.speak_response(answer)  # 응답 텍스트를 음성으로 변환하여 재생

            else:
                print("output.json 형식이 올바르지 않습니다.")  # 오류 메시지 출력

    ######----------yeom_modified----------------######\
    # 음성 서비스가 들어왔을때, True를 출력하는 함수
    def listen_for_voice_service(self):
        if self.voice_to_text_to_voice.audio_playing:  # 음성 녹음을 하여야 하기때문에 해당 음성이 재생중인지 확인
            return False
        if not self.lock.locked():  # 락이 걸려있지 않은 경우에만 실행
            self.voice_to_text_to_voice.play_sound(self.voice_to_text_to_voice.voice_wait)  # 음성 서비스를 대기중이라는 음성을 출력
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("키워드 '음성 서비스'를 대기 중입니다...")
                try:
                    audio = r.listen(source, timeout=10)  # 타임아웃을 10초로 설정
                    text = r.recognize_google(audio, language='ko-KR')
                    print(f"인식된 텍스트: {text}")
                    if ("음성 서비스" in text) or ("음성서비스" in text)or("서비스" in text):
                        print("키워드 '음성 서비스'가 인식되었습니다.")
                        return True
                    else:
                        return False
                except sr.UnknownValueError:
                    print("키워드를 인식할 수 없습니다.")
                    return False
                except sr.RequestError as e:
                    print(f"구글 음성 인식 API 요청에 실패했습니다: {e}")
                    return False
                except sr.WaitTimeoutError:
                    print("키워드 인식 시간 초과.")
                    return False
        else:
            return False
    ######----------yeom_modified----------------######\

    def run(self):
        try:
            while True:  # 무한 루프 시작
                if self.listen_for_voice_service():  # 음성 서비스가 인식되었을 경우만
                    if not self.voice_to_text_to_voice.audio_playing:  # 오디오가 재생 중이지 않은 경우
                        question = self.voice_to_text_to_voice.record_voice()  # 음성을 녹음하여 텍스트로 변환
                        if question:  # 질문이 있는 경우
                            self.question = question
                            self.voice_to_text_to_voice.save_input_json(question)  # 질문을 JSON 파일로 저장
                            self.voice_to_text_to_voice.send_input_json()  # JSON 파일을 서버로 전송
        except KeyboardInterrupt:  # 키보드 인터럽트 발생 시
            self.observer.stop()  # 감시자 중지
        self.observer.join()  # 감시자 종료 대기

if __name__ == "__main__":
    remote_server = "61.108.166.15"  # 원격 서버 주소 설정
    remote_path = "/home/team998/workspace/team2-mainproject/llama-3-Korean-Bllossom-8B"  # 원격 서버 경로 설정
    remote_user = "team998"  # 원격 서버 사용자명 설정
    remote_password = "kcci@123"  # 원격 서버 비밀번호 설정

    assistant = VoiceAssistant("input.json", "output.json", remote_server, remote_path, remote_user, remote_password)  # VoiceAssistant 객체 생성
    assistant.run()  # VoiceAssistant 실행
