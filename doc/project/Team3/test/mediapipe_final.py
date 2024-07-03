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

import cv2
import mediapipe as mp
import numpy as np
import time, os
from tensorflow.keras.models import load_model
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load mp3 files
o_action_music = 'YouTube tabata.mp3'

actions = ['stretch1', 
           'stretch2', 
           'O', 
           'stand', 
           'stand', 
           'jumpingjack', 
           'oneleg',
           'squat',
           'run z'
          ]
seq_length = 30

model = load_model("model/model.keras")

# MediaPipe hands model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

seq = []
action_seq = []
squat_count = 0
squat_detected = False
runz_count = 0
runz_detected = False
o_action_detected = False
music_playing = False

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        
        ret, image = cap.read()
        if not ret:
            print("카메라를 찾을 수 없습니다.")
            # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
            break
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        if result.pose_landmarks is not None:
            
            joint = np.zeros((33, 4))
            for j, lm in enumerate(result.pose_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                
            # Compute angles
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
            seq.append(d)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.9:
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action
            
            if this_action == 'squat':
                if not squat_detected:
                    squat_count += 1
                    squat_detected = True
                    print("squat : ", squat_count)
            else:
                squat_detected = False

            if this_action == 'run z':
                if not runz_detected:
                    runz_count += 1
                    runz_detected = True
                    print("run z : ", runz_count)
            else:
                runz_detected = False
            
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

            cv2.putText(image,
                        f'{this_action.upper()}',
                        org=(int(result.pose_landmarks.landmark[0].x * image.shape[1]),
                        int(result.pose_landmarks.landmark[0].y * image.shape[0] + 20)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(1) == ord('q'):
            break
    

cap.release()
cv2.destroyAllWindows()