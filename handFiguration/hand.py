#!/usr/bin/env python3
# v.1.4
# 손모양 총 15개 구현 완료 / 제시된 손모양을 맞추는 기능 추가

import cv2
import mediapipe as mp
import numpy as np
import random as r
import datetime
from PIL import ImageFont, ImageDraw, Image


#타이머 폰트
timefont = ImageFont.truetype('fonts/SCDream6.otf', 20)

hand_gesture = {
    0:'fist', 1:'one', 2:'gun', 3:'three', 4:'four', 5:'five',
    6:'promise', 7:'spiderman', 8:'niconiconi', 9:'two', 10:'ok',
    11:'claws', 12:'good', 13:'fanxyChild', 14:'dog'
}#게임에 사용할 제스처 세트
input_gesture = 0
input_gesture_switch = 0
delay_time = 0

#MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils #웹캠에서 손가락 뼈마디 부분을 그리는 것
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)  #모드 세팅

#Gesture recognition model
file = np.genfromtxt('C:\git_open_02\Helloworld_A\T03\handFiguration\data\gesture_trained.csv', delimiter=',')    #csv 파일 받아와서 필요한 정보 뽑기
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()  #KNN(K-Nearest Neighbors) 알고리즘을 통해 손모양 학습?
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cam = cv2.VideoCapture(0) #캠켜기
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  #캠크기 조절
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #캠크기 조절

while cam.isOpened():   #카메라가 열려있으면..
    success, image = cam.read() #한 프레임 씩 읽어옴
    
    if not success: #success 못하면 다음 프레임으로..?
        continue
        #success하면 go
        
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) #이미지 전처리(색상 형식 변경 & 이미지 한번 뒤집기)
    results = hands.process(image)  #전처리 및 모델 추론을 함께 실행..
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #출력을 위해 다시 색상 형식 바꿔주기

    if results.multi_hand_landmarks:    #위 전처리를 통해 손이 인식 되면 참이됨
        for hand_landmarks in results.multi_hand_landmarks: #손 여러개 대비?? 예외처리 방지? with 써야되나?
            joint = np.zeros((21, 3))   #joint -> 빨간 점. 포인트 21개, xyz 3개. 생성
            for j, lm in enumerate(hand_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z]   #값 입력
                    
            #joint 인덱스끼리 빼줘서 뼈대의 벡터 구하기(Fig 3의 형태)
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            #벡터의 길이로.. Normalize v?
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                
            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
                    
            angle = np.degrees(angle) # Convert radian to degree
                
            # Inference gesture / 데이터 바꿔주고 정리..
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])
                

            #cv2.putText(image, text = hand_gesture[idx].upper(), org=(20, 60), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = 255, thickness = 3)
            #손 가장 아랫부분에 제스쳐 이름 출력
            org = (int(hand_landmarks.landmark[0].x * image.shape[1]), int(hand_landmarks.landmark[0].y * image.shape[0]))
            cv2.putText(image, text=hand_gesture[idx].upper(), org=(org[0], org[1] + 30), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color=255, thickness = 2)

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS) #마디마디에 그려주는
                
                #손모양이 맞는지 아닌지 맞춰보기
            if input_gesture_switch:
                if idx == input_gesture:
                    delay_time += 1
                    if delay_time > 15:
                        input_gesture_switch = 0
                        delay_time = 0
                        print("okok")


    ret, frame = cam.read()

    # 현재시각을 불러와 문자열로저장
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

    ret, frame = cam.read()     # 카메라로부터 현재 영상을 받아 frame에 저장, 잘 받았다면 ret가 참
    # 글자가 잘보이도록 배경을 넣어줌
    # img는 사각형을 넣을 이미지, pt1, pt2는 사각형의 시작점과 끝점, color는 색상(파랑,초록,빨강), tickness는 선굵기(-1은 내부를 채우는 것)
    # cv2.rectangle(img=frame, pt1=(10, 15), pt2=(340, 35), color=(0,0,0), thickness=-1)     
    
    # 아래의 4줄은 글자를 영상에 더해주는 역할을 함    
    frame = Image.fromarray(frame)    
    draw = ImageDraw.Draw(frame)    
    # xy는 텍스트 시작위치, text는 출력할 문자열, font는 글꼴, fill은 글자색(파랑,초록,빨강)   
    draw.text(xy=(10, 15),  text="rest time "+nowDatetime, font=timefont, fill=(255, 255, 255))
    frame = np.array(frame)

    cv2.imshow("rest time", frame)   # 남은 시간을 표시하는 글자를 써준 영상 출력

           
    if input_gesture_switch == 1:
        cv2.putText(image, text = hand_gesture[input_gesture].upper(), org = (10,30), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = 255, thickness = 2)
        
    if cv2.waitKey(1) == ord('r'):  #input_gesture 랜덤으로 뚱땅뚱땅(임시)
        input_gesture = r.randrange(15)
        input_gesture_switch = 1
        print(input_gesture_switch)

    cv2.imshow('Hand Cam', image)

    if cv2.waitKey(1) == ord('q'):  #q누르면 종료
        break


cam.release()

#안되네요..ㅠㅠ
#
#def random_gesture(input_gesture_switch):
#    input_gesture = r.randrange(15)
#    input_gesture_switch = 1
#    print(input_gesture_switch)
#
#    return input_gesture
#
#def hf_qna(input_gesture_switch, idx, img, delay_time = 0):
#    if input_gesture_switch:
#        if idx == input_gesture:
#            delay_time += 1
#            if answer_delay > 20:
#                input_gesture_switch = 0
#                delay_time = 0
#                cv2.putText(img, text = "Correct", org = (100,240), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255, 0, 0), thickness = 5)
#