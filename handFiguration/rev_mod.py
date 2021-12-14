import cv2
import numpy as np


#함수 부분
def hand_img_revert(image):
    

    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    ret, flag1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    cv2.imshow("BINARY",flag1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


