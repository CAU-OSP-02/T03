#!/usr/bin/env python3

import tkinter as tk
import pygame
import sys
import cv2
import mediapipe as mp
import numpy as np
import random as r
import threading
import datetime
import os
#from PIL import Image
#from PIL import ImageTk
import PIL.Image, PIL.ImageTk
from pygame import mixer
from tkinter import *
from tkinter import ttk
from PyQt5.QtWidgets import QMainWindow, QApplication, QDesktopWidget
from PyQt5.QtCore import Qt
from sys import version_info


#pygame.mixer.init()

#def play():
    #pygame.mixer.music.load('Nautillus.mp3')
    #pygame.mixer.music.play(loops=0)

#def stop():
    #pygame.mixer.music.stop()

def realcam():
    class camcamcam(tk.Frame):
        def camThread():
    #            color = []
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  #캠크기 조절
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            Frame = None

            if (cap.isOpened() == False):
                print("Unable to read camera feed")

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = PIL.Image.fromarray(image)
                image = PIL.ImageTk.PhotoImage(image)

                if Frame is None:
                    Frame = tk.Label(image=image)
                    Frame.image = image
                    Frame.pack(side="top")
                else:
                    Frame.configure(image=image)
                    Frame.image = image

                cv2.waitKey(1)

        if __name__ == '__main__':

            thread_img = threading.Thread(target=camThread, args=())
            thread_img.daemon = True
            thread_img.start()

#            root = tk.Tk()
#            root.title("Hand Figuration")
#            root.geometry('1920x1080')
#            root.minsize(1768,992)
#            root.maxsize(2560,1440)
#            root.mainloop()

    
#class st_cam():

class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self._frame = None
        self.switch_frame(mainmenu)
        #play()

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
           self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

class realstart(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        realcam()
        tk.Label(self, text="").pack()
        tk.Button(self, text="Quit",
                  command=lambda: quit(), font=('휴먼엑스포', 30)).pack()

def cam():
    hand_gesture = {
        0:'fist', 1:'one', 2:'gun', 3:'three', 4:'four', 5:'five',
        6:'promise', 7:'spiderman', 8:'niconiconi', 9:'two', 10:'ok',
        11:'claws', 12:'good', 13:'fanxyChild', 14:'dog'
    }   #게임에 사용할 제스처 세트
    input_gesture = 0
    input_gesture_switch = 0
    delay_time = 0

    #MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils #웹캠에서 손가락 뼈마디 부분을 그리는 것
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)  #모드 세팅

    #Gesture recognition model
    file = np.genfromtxt('C:\\Users\\galax\\OneDrive\\바탕 화면\\T03\\handFiguration\\data\\gesture_trained.csv', delimiter=',')    #csv 파일 받아와서 필요한 정보 뽑기
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
                

    #            cv2.putText(image, text = hand_gesture[idx].upper(), org=(20, 60), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = 255, thickness = 3)
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

class mainmenu(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="Hand Figuration", font=('휴먼엑스포', 60)).pack(side="top", fill="x", pady=150)
        tk.Button(self, text="Start",
                  command=lambda: cam(), width = 10 , height = 1, font=('휴먼엑스포', 30)).pack()
        tk.Label(self, text="").pack()
        #tk.Button(self, text="Store",
                  #command=lambda: master.switch_frame(Store), width = 10 , height = 1, font=('휴먼엑스포', 30)).pack()
        #tk.Label(self, text="").pack()
        tk.Button(self, text="Ranking",
                  command=lambda: master.switch_frame(Ranking), width = 10 , height = 1, font=('휴먼엑스포', 30)).pack()
        tk.Label(self, text="").pack()
        tk.Button(self, text="How To Do",
                  command=lambda: master.switch_frame(How_To_Do), width = 10 , height = 1, font=('휴먼엑스포', 30)).pack()
        tk.Label(self, text="").pack()
        tk.Button(self, text="Setting",
                  command=lambda: master.switch_frame(Setting), width = 10 , height = 1, font=('휴먼엑스포', 30)).pack()
        tk.Label(self, text="").pack()
        tk.Button(self, text="Exit",
                  command=lambda: master.switch_frame(quit), width = 10 , height = 1, font=('휴먼엑스포', 30)).pack()

        def quit(self):
            self.destroy()

class Start(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="Start", font=('휴먼엑스포', 60)).pack(side="top", fill="x", pady=100)
        tk.Button(self, text="시작하기",
                  command=lambda: master.switch_frame(realstart), width = 10 , height = 1, font=('휴먼엑스포', 30)).pack()
        tk.Label(self, text="").pack()
        tk.Button(self, text="Go Back To Main Menu",
                  command=lambda: master.switch_frame(mainmenu), font=('휴먼엑스포', 30)).pack()

'''class Store(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="Store", font=('휴먼엑스포', 60)).pack(side="top", fill="x", pady=100)
        tk.Label(self, text="아이템을 구매하여 더욱 쾌적한 플레이를 즐겨보세요.", font=('휴먼엑스포', 25)).pack()
        tk.Label(self, text="").pack()
        tk.Label(self, text="").pack()
        labelExample = tk.Button(self, text="999999", font=('휴먼엑스포', 30))
        labelExample1 = tk.Button(self, text="0", font=('휴먼엑스포', 20))
        labelExample2 = tk.Button(self, text="0", font=('휴먼엑스포', 20))
        labelExample3 = tk.Button(self, text="0", font=('휴먼엑스포', 20))
        labelExample.pack()
        tk.Label(self, text="").pack()

        def change_label_number1():
            counter = int(str(labelExample['text']))
            counter -= 1000
            labelExample.config(text=str(counter))
            counter = int(str(labelExample1['text']))
            counter += 1
            labelExample1.config(text=str(counter))
        def change_label_number2():
            counter = int(str(labelExample['text']))
            counter -= 2000
            labelExample.config(text=str(counter))
            counter = int(str(labelExample2['text']))
            counter += 1
            labelExample2.config(text=str(counter))
        def change_label_number3():
            counter = int(str(labelExample['text']))
            counter -= 1000
            labelExample.config(text=str(counter))
            counter = int(str(labelExample3['text']))
            counter += 1
            labelExample3.config(text=str(counter))
            
        tk.Button(self, text="방해요소제거 (1000)", width=15, height=1, font=('휴먼엑스포', 20),
                                command=change_label_number1).pack()
        labelExample1.pack()
        tk.Label(self, text="").pack()
        tk.Button(self, text="시간증가 (2000)", width=15, height=1,font=('휴먼엑스포', 20),
                                command=change_label_number2).pack()
        labelExample2.pack()
        tk.Label(self, text="").pack()
        tk.Button(self, text="코인두배 (1000)", width=15, height=1,font=('휴먼엑스포', 20),
                                command=change_label_number3).pack()
        labelExample3.pack()            
        tk.Label(self, text="").pack()
        tk.Button(self, text="Go Back To Main Menu",
                  command=lambda: master.switch_frame(mainmenu), font=('휴먼엑스포', 30)).pack()'''

class Ranking(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="Ranking", font=('휴먼엑스포', 60)).pack(side="top", fill="x", pady=100)
        tk.Button(self, text="Go Back To Main Menu",
                  command=lambda: master.switch_frame(mainmenu), font=('휴먼엑스포', 30)).pack()

class How_To_Do(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="How To Do", font=('휴먼엑스포', 60)).pack(side="top", fill="x", pady=100)
        tk.Label(self, text="1. 화면에 뜨는 손동작에 맞춰 손모양을 취해주세요. 웹캠이 당신의 손바닥을 인식해 성공여부를 판별할 것입니다.", font=('휴먼엑스포', 25)).pack()
        tk.Label(self, text="").pack()
        tk.Label(self, text="2. 다양한 모드를 선택해 색다른 게임플레이를 즐겨보세요.", font=('휴먼엑스포', 25)).pack()
        tk.Label(self, text="").pack()
        tk.Label(self, text="3. 점수표 화면을 통해 친한 친구들과 높은 점수를 노리고 경쟁해 보세요.", font=('휴먼엑스포', 25)).pack()
        tk.Label(self, text="").pack()
        tk.Label(self, text="4. 상점에서 아이템을 구입해 다양한 혜택을 누려보세요.", font=('휴먼엑스포', 25)).pack()
        tk.Label(self, text="").pack()
        tk.Label(self, text="").pack()
        tk.Button(self, text="Go Back To Main Menu",
                  command=lambda: master.switch_frame(mainmenu), font=('휴먼엑스포', 30)).pack()

class Setting(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        app.attributes("-fullscreen", False)
        tk.Label(self, text="Setting", font=('휴먼엑스포', 60)).pack(side="top", fill="x", pady=100)
        tk.Button(self, text="Go Back To Main Menu",
                  command=lambda: master.switch_frame(mainmenu),font=('휴먼엑스포', 30), width = 20 , height = 1).pack()
        tk.Label(self, text="").pack()
        tk.Button(self, text="Switch To Fullscreen",
                  command=lambda: master.switch_frame(Setting1),font=('휴먼엑스포', 30), width = 20 , height = 1).pack()
        #tk.Label(self, text="").pack()
        #tk.Button(self, text="Music On",
                  #command=play,font=('휴먼엑스포', 30), width = 10 , height = 1).pack(side=tk.LEFT)
        #tk.Button(self, text="Music Off",
                  #command=stop,font=('휴먼엑스포', 30), width = 10 , height = 1).pack(side=tk.RIGHT)


class Setting1(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        app.attributes("-fullscreen", True)
        tk.Label(self, text="Setting", font=('휴먼엑스포', 60)).pack(side="top", fill="x", pady=100)
        tk.Button(self, text="Go Back To Main Menu",
                  command=lambda: master.switch_frame(mainmenu), font=('휴먼엑스포', 30), width = 20 , height = 1).pack()
        tk.Label(self, text="").pack()
        tk.Button(self, text="Switch To Window",
                  command=lambda: master.switch_frame(Setting),font=('휴먼엑스포', 30), width = 20 , height = 1).pack()
        #tk.Label(self, text="").pack()
        #tk.Button(self, text="Music On",
                  #command=play,font=('휴먼엑스포', 30), width = 10 , height = 1).pack(side=tk.LEFT)
        #tk.Button(self, text="Music Off",
                  #command=stop,font=('휴먼엑스포', 30), width = 10 , height = 1).pack(side=tk.RIGHT)


if __name__ == "__main__":
    app = SampleApp()
    app.title("Hand Figuration")
    app.geometry('1920x1080')
    app.minsize(1768,992)
    app.maxsize(2560,1440)
    app.mainloop()
