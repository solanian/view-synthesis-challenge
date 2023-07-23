import math
import cv2
import numpy as np
img = cv2.imread("../content/1.png")



num = 5
if num == 1:
    # parameters description :  https://bskyvision.com/entry/%EB%AF%B8%EB%94%94%EC%96%B4%EC%BF%BC%EB%A6%AC%EA%B0%80-%EC%8A%A4%EB%A7%88%ED%8A%B8%ED%8F%B0%EC%97%90%EC%84%9C-%EC%A0%9C%EB%8C%80%EB%A1%9C-%EC%A0%81%EC%9A%A9%EB%90%98%EC%A7%80-%EC%95%8A%EC%9D%84%EB%95%8C-%EC%B2%B4%ED%81%AC%ED%95%A0-%EA%B2%83
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
elif num == 2:
    # https://bskyvision.com/24
    dst = cv2.GaussianBlur(img, (5,5),0)
elif num == 3:
    # https://bskyvision.com/24
    dst = cv2.medianBlur(img, 5)
elif num == 4:
    # https://bskyvision.com/24
    dst = cv2.bilateralFilter(img, 5, 50, 50)
# cv2.imwrite("../content/1_res.png",dst)
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow(winname='image', width=1800, height=1230)
cv2.imshow('image', np.hstack((img, dst)))
cv2.waitKey(0)  # 키보드 입력을 대기하는 함수, milisecond값을 넣으면 해당 시간동안 대기, 0인경우 무한으로 대기
cv2.destoryAllWindows()  # 표시했던 윈도우를 종료
