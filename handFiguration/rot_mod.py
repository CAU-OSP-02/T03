import cv2 
#함수 부분
def hand_img_rotation(img, degree): 
    height, width = img.shape[:-1] 
    centerRotatePT = int(width / 2), int(height / 2) 
    rotatefigure = cv2.getRotationMatrix2D(centerRotatePT, degree, 1) 
    result = cv2.warpAffine(img, rotatefigure, (height, width)) 
    cv2.imshow('img', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







