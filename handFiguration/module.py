import cv2
import numpy as np


#함수 부분
def hand_img_revert(image):
    

    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    ret, flag1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    return flag1


def hand_img_rotation(img, degree): 
    height, width = img.shape[:-1] 
    centerRotatePT = int(width / 2), int(height / 2) 
    rotatefigure = cv2.getRotationMatrix2D(centerRotatePT, degree, 1) 
    result = cv2.warpAffine(img, rotatefigure, (height, width)) 
    
    return result

