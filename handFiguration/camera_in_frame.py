from PIL import Image
from PIL import ImageTk
import mediapipe as mp
import numpy as np
import tkinter as tk
import threading
import datetime
import cv2
import os

def camcam():
    def camThread():
        color = []
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  #캠크기 조절
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        panel = None

        if (cap.isOpened() == False):
            print("Unable to read camera feed")

        while True:
            ret, color = cap.read()
            if (color != []):

                image = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                if panel is None:
                    panel = tk.Label(image=image)
                    panel.image = image
                    panel.pack(side="left")
                else:
                    panel.configure(image=image)
                    panel.image = image

                cv2.waitKey(1)

    if __name__ == '__main__':

        thread_img = threading.Thread(target=camThread, args=())
        thread_img.daemon = True
        thread_img.start()

        root = tk.Tk()
        root.title("Hand Figuration")
        root.geometry('1920x1080')
        root.minsize(1768,992)
        root.maxsize(2560,1440)
        root.mainloop()

camcam()