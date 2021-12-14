import cv2
import rev_mod  #색 반전 함수
import rot_mod   #사진 회전 함수


image = 'C:/Python_files/image/1.jpg'               #이미지 경로
print(rev_mod.hand_img_revert(image))                          
                                                               
image = cv2.imread('C:/Python_files/image/1.jpg')   #이미지 경로
print(rot_mod.hand_img_rotation(image, 90))




