import cv2 as cv
import mediapipe as mp
import time
import math


class PoseDetector():
    def __init__(self,mode=False,upBody=False,smooth=True,detectioncon=0.5,trackingcon=0.5):
        self.mode = mode
        self.upBody=upBody
        self.smooth=smooth
        self.detectioncon=detectioncon
        self.trackingcon=trackingcon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()

        self.mpDraw = mp.solutions.drawing_utils
    

    def findpose(self,img,draw=False):
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)

        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

        return img
    
    def findposition(self,img,draw=False):
        self.lmlist = []
        if self.results.pose_landmarks:
            
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                self.lmlist.append([id,cx,cy])
                if draw:
                    self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
                    cv.circle(img,(cx,cy),10,(255,0,255),4)


        return self.lmlist
    
    def findAngle(self,img,pt1,pt2,pt3,draw=True):
        x1,y1 = self.lmlist[pt1][1:]
        x2,y2 = self.lmlist[pt2][1:]
        x3,y3 = self.lmlist[pt3][1:]

        angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))

        if angle<0:
            angle += 360

        if draw:
            cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv.line(img,(x3,y3),(x2,y2),(0,0,255),2)
            cv.circle(img,(x1,y1),10,(0,0,255),cv.FILLED)
            cv.circle(img,(x1,y1),5,(0,0,255),2)
            cv.circle(img,(x2,y2),10,(0,0,255),cv.FILLED)
            cv.circle(img,(x2,y2),5,(0,0,255),2)
            cv.circle(img,(x3,y3),10,(0,0,255),cv.FILLED)
            cv.circle(img,(x3,y3),5,(0,0,255),2)

            cv.putText(img,str(int(angle)),(x2-50,y2+50),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

        return angle




def main():
    cap = cv.VideoCapture(r"C:\Users\HP\Downloads\8402080-hd_1080_1920_30fps.mp4")
    detector = PoseDetector()
    ptime = 0
    while True:
        success,img = cap.read()
        if not success:
            break

        img = cv.resize(img,None,fx=0.5,fy=0.5,interpolation=cv.INTER_LINEAR)

        img = detector.findpose(img)
        lmlist = detector.findposition(img)

        if len(lmlist) != 0:
            cv.circle(img,(lmlist[13][1],lmlist[13][2]),10,(0,0,255),3,cv.FILLED)
        
        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime
        cv.putText(img,str(int(fps)),(50,70),cv.FONT_HERSHEY_COMPLEX,3,(255,0,255),2)

        cv.imshow("Image",img)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()