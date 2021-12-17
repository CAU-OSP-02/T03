#!/usr/bin/env python3

from tkinter import *
import os

File = 'data/hand_pic'
file_list = os.listdir(File)
real_file_list = [x for x in file_list if(x.endswith(".JPEG") or (x.endswith(".jpeg")==True))]
print(real_file_list)

xn=0
root=Tk()
root.title("Source Image")
root.geometry("500x500")
root.resizable(0, 0)
idx = r.randrange(14)

img = PIL.ImageTk.PhotoImage(PIL.Image.open('./T03/handFiguration/data/hand_pic/'+ str(idx) +'.jpeg'))
label = Label(image=img)
label.pack()

n = Button(root, text='ran', command=)
n.pack()

quit = Button(root, text='종료하기', command=root.quit)
quit.pack()

print(xn)
root.mainloop()
