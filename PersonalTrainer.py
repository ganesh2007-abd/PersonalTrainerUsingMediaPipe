import cv2 as cv
import mediapipe as mp
import numpy as np
import PoseModule as pm
import time

cap = cv.VideoCapture(r"D:\Projects\PersonalTrainerUsingMediaPipe\bicep-curl_ErbbKGjW.mp4")
detector = pm.PoseDetector()
count=0
dir=0

while True:
    bool,img = cap.read()
    if not bool:
        break

    img = cv.resize(img,None,fx=0.15,fy=0.15,interpolation=cv.INTER_AREA)

    h,w,c = img.shape
    print(w,h)

    img = detector.findpose(img,draw=False)
    lmlist = detector.findposition(img,draw=False)

    if len(lmlist) != 0:
        angle=detector.findAngle(img,11,13,15)

        per = np.interp(angle,(210,310),(0,100))
        bar = np.interp(angle,(220,310),(650,100))

        color = (255,0,255)

        if per>=99:
            color = (0,255,0)
            if dir == 0:
                count += 0.5
                dir = 1

        if per<=1:
            color=(0,255,0)
            if dir == 1:
                count += 0.5
                dir = 0

        cv.rectangle(img,(300,20),(320,570),color,3)
        cv.rectangle(img,(300,int(bar)-80),(320,570),color,cv.FILLED)

        cv.putText(img,str(int(per)),(250,40),cv.FONT_HERSHEY_COMPLEX,0.75,(255,0,0),2)
        cv.putText(img,str(int(count)),(30,100),cv.FONT_HERSHEY_DUPLEX,3,(0,0,255),2)


    cv.imshow("Image",img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()