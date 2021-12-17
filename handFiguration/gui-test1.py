#!/usr/bin/env python3

from tkinter import *
#import tkinter as tk
#from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
import PIL.Image, PIL.ImageTk
import threading
import random as r
#import os
#import sys
#import datetime
#from PIL import Image
#from PIL import ImageTk
from pygame import mixer
import pygame
#from PyQt5.QtWidgets import QMainWindow, QApplication, QDesktopWidget
#from PyQt5.QtCore import Qt
#from sys import version_info


pygame.mixer.init()

def play():
    pygame.mixer.music.load('Nautillus.mp3')
    pygame.mixer.music.play(loops=0)

def stop():
    pygame.mixer.music.stop()

hand_gesture = {
    0:'fist', 1:'one', 2:'gun', 3:'three', 4:'four', 5:'five',
    6:'promise', 7:'spiderman', 8:'niconiconi', 9:'two', 10:'ok',
    11:'claws', 12:'good', 13:'fanxyChild', 14:'dog'
}   #게임에 사용할 제스처 세트
input_gesture = 0
input_gesture_switch = 0
delay_time = 0
cam = None

#MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils #웹캠에서 손가락 뼈마디 부분을 그리는 것
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)  #모드 세팅

#Gesture recognition model
file = np.genfromtxt('data/gesture_trained.csv', delimiter=',')    #csv 파일 받아와서 필요한 정보 뽑기
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()  #KNN(K-Nearest Neighbors) 알고리즘을 통해 손모양 학습?
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

def realcam():
    class camcamcam(Frame):
        global cam
        def camThread():
            cam = cv2.VideoCapture(0) #캠켜기
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 650)  #캠크기 조절
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #캠크기 조절
            Frame = None

            while cam.isOpened():   #카메라가 열려있으면..
                success, image = cam.read() #한 프레임 씩 읽어옴
                if not success: #success 못하면 다음 프레임으로..?
                    continue
                #success하면 go
                
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) #이미지 전처리(색상 형식 변경 & 이미지 한번 뒤집기)
                results = hands.process(image)  #전처리 및 모델 추론을 함께 실행..
                image = PIL.Image.fromarray(image)
                image = PIL.ImageTk.PhotoImage(image)

                if Frame is None:
                    Frame = Label(image=image)
                    Frame.image = image
                    Frame.pack(side="top")
                else:
                    Frame.configure(image=image)
                    Frame.image = image
                    

        if __name__ == '__main__':
                thread_img = threading.Thread(target=camThread, args=())
                thread_img.daemon = True
                thread_img.start()

class SampleApp(Tk):
    def __init__(self):
        Tk.__init__(self)
        self._frame = None
        self.switch_frame(mainmenu)
        #play()

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
           self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

class mainmenu(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        Label(self, text="Hand Figuration", font=('휴먼엑스포', 50)).pack(side="top", fill="x", pady=70)
        Button(self, text="Start",
                  command=lambda: master.switch_frame(Start), width = 10 , height = 1, font=('휴먼엑스포', 25)).pack()
        Label(self, text="").pack()
        Button(self, text="Store",
                  command=lambda: master.switch_frame(Store), width = 10 , height = 1, font=('휴먼엑스포',25)).pack()
        Label(self, text="").pack()
        Button(self, text="Ranking",
                  command=lambda: master.switch_frame(Ranking), width = 10 , height = 1, font=('휴먼엑스포',25)).pack()
        Label(self, text="").pack()
        Button(self, text="How To Do",
                  command=lambda: master.switch_frame(How_To_Do), width = 10 , height = 1, font=('휴먼엑스포',25)).pack()
        Label(self, text="").pack()
        Button(self, text="Setting",
                  command=lambda: master.switch_frame(Setting), width = 10 , height = 1, font=('휴먼엑스포',25)).pack()
        Label(self, text="").pack()
        Button(self, text="Exit",
                  command=lambda: master.switch_frame(quit), width = 10 , height = 1, font=('휴먼엑스포',25)).pack()

        def quit(self):
            self.destroy()

class Start(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        realcam()
        Label(self, text="").pack()
        Button(self, text="Quit",
                  command=lambda: quit(), font=('휴먼엑스포', 25)).pack(side='bottom')

class Store(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        Label(self, text="Store", font=('휴먼엑스포',50)).pack(side="top", fill="x", pady=30)
        Label(self, text="아이템을 구매하여 더욱 쾌적한 플레이를 즐겨보세요.", font=('휴먼엑스포',20)).pack()
        Label(self, text="").pack()
        Label(self, text="").pack()
        labelExample = Button(self, text="999999", font=('휴먼엑스포',25))
        labelExample1 = Button(self, text="0", font=('휴먼엑스포', 20))
        labelExample2 = Button(self, text="0", font=('휴먼엑스포', 20))
        labelExample3 = Button(self, text="0", font=('휴먼엑스포', 20))
        labelExample.pack()
        Label(self, text="").pack()

        def change_label_number1():
            counter = int(str(labelExample['text']))
            counter -= 700
            labelExample.config(text=str(counter))
            counter = int(str(labelExample1['text']))
            counter += 1
            labelExample1.config(text=str(counter))
        def change_label_number2():
            counter = int(str(labelExample['text']))
            counter -= 2500
            labelExample.config(text=str(counter))
            counter = int(str(labelExample2['text']))
            counter += 1
            labelExample2.config(text=str(counter))
        def change_label_number3():
            counter = int(str(labelExample['text']))
            counter -= 700
            labelExample.config(text=str(counter))
            counter = int(str(labelExample3['text']))
            counter += 1
            labelExample3.config(text=str(counter))
            
        Button(self, text="방해요소제거 (700)", width=15, height=1, font=('휴먼엑스포', 20),
                                command=change_label_number1).pack()
        labelExample1.pack()
        Label(self, text="").pack()
        Button(self, text="시간증가 (2500)", width=15, height=1,font=('휴먼엑스포', 20),
                                command=change_label_number2).pack()
        labelExample2.pack()
        Label(self, text="").pack()
        Button(self, text="코인두배 (700)", width=15, height=1,font=('휴먼엑스포', 20),
                                command=change_label_number3).pack()
        labelExample3.pack()            
        Label(self, text="").pack()
        Button(self, text="Go Back To Main Menu",
                  command=lambda: master.switch_frame(mainmenu), font=('휴먼엑스포',20)).pack()

class Ranking(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        Label(self, text="Ranking", font=('휴먼엑스포',50)).pack(side="top", fill="x", pady=70)
        Button(self, text="Go Back To Main Menu",
                  command=lambda: master.switch_frame(mainmenu), font=('휴먼엑스포',20)).pack()

class How_To_Do(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        Label(self, text="How To Do", font=('휴먼엑스포',50)).pack(side="top", fill="x", pady=70)
        Label(self, text="1. 화면에 뜨는 손동작에 맞춰 손모양을 취해주세요. 웹캠이 당신의 손바닥을 인식해 성공여부를 판별할 것입니다.", font=('휴먼엑스포',18)).pack()
        Label(self, text="").pack()
        Label(self, text="2. 다양한 모드를 선택해 색다른 게임플레이를 즐겨보세요.", font=('휴먼엑스포',18)).pack()
        Label(self, text="").pack()
        Label(self, text="3. 점수표 화면을 통해 친한 친구들과 높은 점수를 노리고 경쟁해 보세요.", font=('휴먼엑스포',18)).pack()
        Label(self, text="").pack()
        Label(self, text="4. 상점에서 아이템을 구입해 다양한 혜택을 누려보세요.", font=('휴먼엑스포',18)).pack()
        Label(self, text="").pack()
        Label(self, text="").pack()
        Button(self, text="Go Back To Main Menu",
                  command=lambda: master.switch_frame(mainmenu), font=('휴먼엑스포',20)).pack()

class Setting(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        app.attributes("-fullscreen", False)
        Label(self, text="Setting", font=('휴먼엑스포',50)).pack(side="top", fill="x", pady=70)
        Button(self, text="Go Back To Main Menu",
                  command=lambda: master.switch_frame(mainmenu),font=('휴먼엑스포',20), width = 25 , height = 1).pack()
        Label(self, text="").pack()
        Button(self, text="Switch To Fullscreen",
                  command=lambda: master.switch_frame(Setting1),font=('휴먼엑스포',20), width = 25 , height = 1).pack()
        Label(self, text="").pack()
        Button(self, text="Music On",
                  command=play,font=('휴먼엑스포',20), width = 12 , height = 1).pack(side=LEFT)
        Button(self, text="Music Off",
                  command=stop,font=('휴먼엑스포',20), width = 12 , height = 1).pack(side=RIGHT)


class Setting1(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        app.attributes("-fullscreen", True)
        Label(self, text="Setting", font=('휴먼엑스포',50)).pack(side="top", fill="x", pady=70)
        Button(self, text="Go Back To Main Menu",
                  command=lambda: master.switch_frame(mainmenu), font=('휴먼엑스포',20), width = 25 , height = 1).pack()
        Label(self, text="").pack()
        Button(self, text="Switch To Window",
                  command=lambda: master.switch_frame(Setting),font=('휴먼엑스포',20), width = 25 , height = 1).pack()
        Label(self, text="").pack()
        Button(self, text="Music On",
                  command=play,font=('휴먼엑스포',20), width = 12 , height = 1).pack(side=LEFT)
        Button(self, text="Music Off",
                  command=stop,font=('휴먼엑스포',20), width = 12 , height = 1).pack(side=RIGHT)


if __name__ == "__main__":
    app = SampleApp()
    app.title("Hand Figuration")
    app.geometry('1366x768')
    app.minsize(1366,768)
    app.maxsize(3840,2160)
    app.mainloop()
