import tkinter as tk
import pygame
import sys
from pygame import mixer
from tkinter import *
from tkinter import ttk
from PyQt5.QtWidgets import QMainWindow, QApplication, QDesktopWidget
from PyQt5.QtCore import Qt
from sys import version_info

pygame.mixer.init()

def play():
    pygame.mixer.music.load("C:/Users/galax/Downloads/Nautillus.mp3")
    pygame.mixer.music.play(loops=0)

def stop():
    pygame.mixer.music.stop()

class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self._frame = None
        self.switch_frame(mainmenu)
        play()

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
           self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

class mainmenu(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="Hand Figuration", font=('휴먼엑스포', 60)).pack(side="top", fill="x", pady=150)
        tk.Button(self, text="Start",
                  command=lambda: master.switch_frame(Start), width = 10 , height = 1, font=('휴먼엑스포', 30)).pack()
        tk.Label(self, text="").pack()
        tk.Button(self, text="Store",
                  command=lambda: master.switch_frame(Store), width = 10 , height = 1, font=('휴먼엑스포', 30)).pack()
        tk.Label(self, text="").pack()
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
        tk.Button(self, text="Go Back To Main Menu",
                  command=lambda: master.switch_frame(mainmenu), font=('휴먼엑스포', 30)).pack()

class Store(tk.Frame):
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
                  command=lambda: master.switch_frame(mainmenu), font=('휴먼엑스포', 30)).pack()

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
        tk.Label(self, text="").pack()
        tk.Button(self, text="Music On",
                  command=play,font=('휴먼엑스포', 30), width = 10 , height = 1).pack(side=tk.LEFT)
        tk.Button(self, text="Music Off",
                  command=stop,font=('휴먼엑스포', 30), width = 10 , height = 1).pack(side=tk.RIGHT)


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
        tk.Label(self, text="").pack()
        tk.Button(self, text="Music On",
                  command=play,font=('휴먼엑스포', 30), width = 10 , height = 1).pack(side=tk.LEFT)
        tk.Button(self, text="Music Off",
                  command=stop,font=('휴먼엑스포', 30), width = 10 , height = 1).pack(side=tk.RIGHT)


if __name__ == "__main__":
    app = SampleApp()
    app.title("Hand Figuration")
    app.geometry('1920x1080')
    app.minsize(1768,992)
    app.maxsize(2560,1440)
    app.mainloop()