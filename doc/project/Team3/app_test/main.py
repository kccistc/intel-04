#!/usr/bin/env python
# coding: utf-8

''' ####label 기억용 개인간 차이 있음
# 0 : 스트레칭 1번 팔 당기기
  1 : 스트레칭 2번 옆구리
  2 : O 표시
  3 : 가만히 있기
  4 : 가만히 있기
  5 : 팔벌려뛰기
  6 : 한 발 서기
  7 : 스쿼트
  8 : 런지
'''

import tkinter as tk
from tkinter import ttk
from ttkbootstrap import Style
from tkinter import messagebox , Toplevel
from PIL import Image, ImageTk
import json
from numpy.lib.stride_tricks import as_strided
import cv2
import mediapipe as mp
import numpy as np
import time, os
from tensorflow.keras.models import load_model
import pygame


window_width = 1600
window_height = 1200
pad = 50
current_dir = os.path.dirname(__file__)

model_path = os.path.join(current_dir,"resource","model")
pose_model = os.path.join(model_path,"human-pose-estimation-0001.xml")

image_path = os.path.join(current_dir,"resource","image")
squart_path = os.path.join(image_path,"squart.png")

filename = 'user_list.json'
cam_check = False


######pose set up


actions = ['stretch1', 
           'stretch2', 
           'O' , 
           'stand' , 
           'stand' , 
           'jumpingjack', 
           'oneleg',
           'squat',
           'lunge'
          ]
seq_length = 30

model = load_model("resource/model/model.keras")

# MediaPipe hands model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Initialize pygame mixer
pygame.mixer.init()

# Load mp3 files
o_action_music = 'YouTube tabata.mp3'

def show_home():
    hide_all_frames()
    home_frame.pack(fill="both", expand=True)
    update_nav_style(home_link)

def show_menu1():
    if not username:
        return
    hide_all_frames()
    menu1_frame.pack(fill="both", expand=True)
    update_nav_style(menu1_link)
    video_capture.start()
    open_new_window(squart_path)

def show_menu2():
    if not username:
        return
    hide_all_frames()
    menu2_frame.pack(fill="both", expand=True)
    update_nav_style(menu2_link)

def show_menu3():
    if not username:
        return
    hide_all_frames()
    menu3_frame.pack(fill="both", expand=True)
    update_nav_style(menu3_link)

def show_about():
    if not username:
        return
    hide_all_frames()
    about_frame.pack(fill="both", expand=True)
    update_nav_style(about_link)

def hide_all_frames():
    home_frame.pack_forget()
    menu1_frame.pack_forget()
    menu2_frame.pack_forget()
    menu3_frame.pack_forget()
    about_frame.pack_forget()
    if cam_check:
        video_capture.stop()
    disable_widgets(home_frame)
    disable_widgets(menu1_frame)
    disable_widgets(menu2_frame)
    disable_widgets(menu3_frame)
    disable_widgets(about_frame)
    
    global current_image_window
    if current_image_window:
        current_image_window.destroy()
        current_image_window = None
    

def disable_widgets(frame):
    for child in frame.winfo_children():
        child.configure(state='disabled')

def confirm_username(event=None):
    global username, filename
    username = username_entry.get().strip()
    if username:
        username_label.grid_forget()
        username_entry.grid_forget()
        confirm_button.grid_forget()
        update_username_display()
        center_frame.pack_forget()
        
        try:
            with open(filename, 'r') as f:
                user_list = json.load(f)
        except json.JSONDecodeError:
            user_list = {}

        # 새로운 사용자 추가
        if username not in user_list.keys():
            user_list[username] = {
                "squat": 0,
                "lunge": 0,
                "total_count": 0
            }

        with open(filename, 'w') as f:
            json.dump(user_list, f, indent=2)
            
    else:
        messagebox.showwarning("Warning", "Please enter a value!") 

def logout_username(event=None):
    global username, username_display, logout_button, username_label, username_entry, confirm_button
    username = None
    username_display.pack_forget()
    logout_button.pack_forget()
    show_home()
    center_frame.pack(pady=150)
    username_label.grid(row=0, column=0, pady=10)
    username_entry.grid(row=1, column=0, pady=10)
    confirm_button.grid(row=2, column=0, pady=10)
    

def update_username_display():
    global username_display, logout_button
    username_display = ttk.Label(collapse, text=f"Welcome,  {username}!", style='Primary.TLabel', font=('Roboto', 12, 'normal'))
    username_display.pack(side="left", padx=10)
    logout_button = ttk.Button(collapse, text="Logout", command=logout_username)
    logout_button.pack(side="left", padx=10)

def update_nav_style(label):
    for link in [home_link, menu1_link, menu2_link, menu3_link, about_link]:
        link.config(font=('Roboto', 12, 'normal'))
    
    label.config(font=('Roboto', 12, 'bold'))

def open_new_window(image_path):
    global current_image_window
    
    if current_image_window:
        current_image_window.destroy()
    
    new_window = Toplevel(window)
    new_window.title("가이드")
    new_window.geometry(f"-50+100")
    # 이미지 로드 및 라벨에 설정
    image = Image.open(image_path)
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(new_window, image=photo)
    label.image = photo  # 이 줄을 추가하여 이미지가 가비지 컬렉션되지 않도록 방지
    label.pack(padx=20, pady=20)
    
    # 현재 이미지 창 업데이트
    current_image_window = new_window

def create_back_button(frame,command):
    button = tk.Button(frame, text="뒤로", command=command)
    button.pack(pady=10)

class VideoCapture:
    def __init__(self, canvas, x, y, width, height):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.cap = None
        self.image_id = None
        self.is_running = False
        self.seq = []
        self.action_seq = []
        self.squat_count = 0
        self.squat_detected = False
        self.lunge_count = 0
        self.lunge_detected = False
        self.o_action_detected = False
        self.music_playing = False
        self.user_list = None
        
    def start(self):
        global cam_check
        cam_check = True
        self.cap = cv2.VideoCapture(0)  # 0번 카메라(기본 웹캠) 사용
        self.is_running = True
        created_time = int(time.time())
        os.makedirs('dataset', exist_ok=True)
        with open(filename, 'r') as f:
            self.user_list = json.load(f)
        self.update()

    def stop(self):
        self.user_list[username]["squat"] = self.squat_count
        self.user_list[username]["lunge"] = self.lunge_count
        self.user_list[username]["total_count"] += self.squat_count + self.lunge_count
        with open(filename, 'w') as f:
            json.dump(self.user_list, f, indent=2)
        print(self.user_list)
        self.is_running = False
        self.cap.release()

    def update(self):
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        frame = frame[:,1*self.width//4 :3*self.width//4]

        if ret:
            self.current_frame = frame.copy()  # 현재 프레임을 변수에 저장
            
            with mp_pose.Pose(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as pose:
                
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(frame)
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                mp_drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                if result.pose_landmarks is not None:
                    joint = np.zeros((33, 4))
                    for j, lm in enumerate(result.pose_landmarks.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                        
                    # Compute an0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25    
                    v1 = joint[[11,11,11,12,12,13,14,15,15,15,16,16,16,17,18,23,23,24,25,26,27,27,28,28,29,30], :3] # Parent joint
                    v2 = joint[[12,13,23,14,24,15,16,17,19,21,18,20,22,19,20,24,25,26,27,28,29,31,30,32,31,32], :3] # Child joint
                    v = v2 - v1 # [26, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,0,0,0,1,1,3,5,5,5, 6, 6, 6,7,7, 7,8, 8,10,10,10,11,11,15,15,16,17,18,18,19,19,20,20,21,22,22,23 ],:], 
                    v[[1,2,3,4,2,5,6,7,8,9,10,11,12,8,9,13,9,13,11,12,14,12,14,16,17,18,19,20,21,22,23,21,24,24,23,25,25 ],:])) 

                    angle = np.degrees(angle) # Convert radian to degree

                    d = np.concatenate([joint.flatten(), angle])
                    self.seq.append(d)

                    if len(self.seq) < seq_length:
                        pass
                    else:
                        input_data = np.expand_dims(np.array(self.seq[-seq_length:], dtype=np.float32), axis=0)
                        y_pred = model.predict(input_data).squeeze()

                        i_pred = int(np.argmax(y_pred))
                        conf = y_pred[i_pred]

                        if conf < 0.9:
                            pass
                        else:
                            action = actions[i_pred]
                            self.action_seq.append(action)

                            if len(self.action_seq) < 3:
                                pass
                            else:
                                this_action = '?'
                                if self.action_seq[-1] == self.action_seq[-2] == self.action_seq[-3]:
                                    this_action = action
                                
                                if this_action == 'squat':
                                    if not self.squat_detected:
                                        self.squat_count += 1
                                        self.squat_detected = True
                                        print("squat : ", self.squat_count)
                                else:
                                    self.squat_detected = False

                                if this_action == 'lunge':
                                    if not self.lunge_detected:
                                        self.lunge_count += 1
                                        self.lunge_detected = True
                                        print("lunge : ", self.lunge_count)
                                else:
                                    self.lunge_detected = False
                                
                                if this_action == 'one leg' :
                                    start_time = time.time()
                                    print("one leg : ", start_time)
                                    fin_time = time.time()
                                    
                                if this_action == 'O':
                                    if not o_action_detected:
                                        o_action_detected = True
                                        if not music_playing:
                                            pygame.mixer.music.load(o_action_music)
                                            pygame.mixer.music.play()
                                            music_playing = True
                                        else:
                                            pygame.mixer.music.stop()
                                            music_playing = False
                                        print("O detected")
                                else:
                                    o_action_detected = False

                                cv2.putText(frame,
                                            f'{this_action.upper()}',
                                            org=(int(result.pose_landmarks.landmark[0].x * frame.shape[1]),
                                            int(result.pose_landmarks.landmark[0].y * frame.shape[0] + 20)),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
            frame = cv2.resize(frame, (self.width, self.height))  # 프레임 크기 조정
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            if self.image_id:
                self.canvas.delete(self.image_id)
            self.image_id = self.canvas.create_image(self.x, self.y, anchor='nw', image=imgtk)
            self.canvas.image = imgtk

        self.canvas.after(10, self.update)  # 10ms마다 업데이트
        
# tkinter 윈도우 생성
window = tk.Tk()
window.title("Great Detective Holmes")
window.geometry(f"{window_width}x{window_height}+100+100")

style = Style(theme='minty')

##### 시작 모드 프레임 생성
home_frame = ttk.Frame(window)
home_frame.pack(fill="x")

# 로고
logo_img = tk.PhotoImage(file="resource/image/logo.png")
logo = ttk.Label(home_frame, image=logo_img)
logo.grid(row=0, column=0, padx=10)

collapse = ttk.Frame(home_frame)
collapse.grid(row=0, column=2, padx=10)

nav = ttk.LabelFrame(collapse, style='secondary.TLabelFrame')
nav.pack(side="left", padx=10, pady=10)

# 메뉴 링크
home_link = ttk.Label(nav, text="Home", style='secondary.TLabel', cursor="hand2")
home_link.grid(row=0, column=0, padx=10)
home_link.bind("<Button-1>", lambda event: show_home())

menu1_link = ttk.Label(nav, text="Menu1", style='secondary.TLabel', cursor="hand2")
menu1_link.grid(row=0, column=1, padx=10)
menu1_link.bind("<Button-1>", lambda event: show_menu1())

menu2_link = ttk.Label(nav, text="Menu2", style='secondary.TLabel', cursor="hand2")
menu2_link.grid(row=0, column=2, padx=10)
menu2_link.bind("<Button-1>", lambda event: show_menu2())

menu3_link = ttk.Label(nav, text="Menu3", style='secondary.TLabel', cursor="hand2")
menu3_link.grid(row=0, column=3, padx=10)
menu3_link.bind("<Button-1>", lambda event: show_menu3())

about_link = ttk.Label(nav, text="About", style='secondary.TLabel', cursor="hand2")
about_link.grid(row=0, column=4, padx=10)
about_link.bind("<Button-1>", lambda event: show_about())

# 입력 필드 생성
center_frame = ttk.Frame(window)
center_frame.pack(pady=150)

username_label = ttk.Label(center_frame, text="Enter your username:")
username_label.grid(row=0, column=0, pady=10)

username_entry = ttk.Entry(center_frame)
username_entry.grid(row=1, column=0, pady=10)

confirm_button = ttk.Button(center_frame, text="Confirm", command=confirm_username)
confirm_button.grid(row=2, column=0, pady=10)

# Enter 키로도 입력 확인 가능하게 설정 
username_entry.bind("<Return>", confirm_username)
#####

##### 메인 메뉴의 프레임 생성
home_frame = ttk.Frame(window)

username_display = None  # 사용자명을 표시할 Label

##### menu1
menu1_frame = ttk.Frame(window)
label_menu1 = ttk.Label(menu1_frame, text="Menu1 Page", font=('Roboto', 18))
label_menu1.pack(pady=20, side='bottom')

video_canvas = tk.Canvas(menu1_frame, width=window_width, height=window_height)
video_canvas.pack(fill="both", expand=True)
video_capture = VideoCapture(video_canvas, window_width // 4,0, window_width // 2, window_height) ## 화면 오른쪽에 refernce image 추가
current_image_window = None
#####


##### menu2
menu2_frame = ttk.Frame(window)
label_menu2 = ttk.Label(menu2_frame, text="Menu2 Page", font=('Roboto', 18))
label_menu2.pack(pady=20, side='bottom')
#####


##### menu3
menu3_frame = ttk.Frame(window)
label_menu3 = ttk.Label(menu3_frame, text="Menu3 Page", font=('Roboto', 18))
label_menu3.pack(pady=20, side='bottom')
#####


##### menu4
about_frame = ttk.Frame(window)
label_about = ttk.Label(about_frame, text="About Page", font=('Roboto', 18))
label_about.pack(pady=20, side='bottom')
#####

show_home()

# tkinter 윈도우 실행
window.mainloop()