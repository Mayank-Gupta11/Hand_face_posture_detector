import cv2
import mediapipe as mp
import time
import math
class posedetect():

    def __init__(self,mode=False, upBody=False,complexity=1, smooth=True, detect_conf=0.5, track_conf=0.5):
        self.mode=mode
        self.upBody=upBody
        self.smooth=smooth
        self.detect_conf=detect_conf
        self.track_conf=track_conf
        self.complexity = complexity
        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose=self.mpPose.Pose(self.mode,self.complexity,self.upBody,self.smooth,self.detect_conf,self.track_conf)

    def findpose(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        result=self.pose.process(imgRGB)
        self.checker=(result.pose_landmarks)
    
        if self.checker:
            if draw:
                self.mpDraw.draw_landmarks(img,self.checker,self.mpPose.POSE_CONNECTIONS)
        
        return img
    

    def findposition(self,img,draw=True):
        self.lmlist=[]
        if self.checker:
            for id,landmrk in enumerate(self.checker.landmark):
                    #h,w,c=img.shape

                    #if id==0:
                    h,w,c=img.shape
                    cx,cy=(int(landmrk.x*w),int(landmrk.y*h))
                    if draw:
                        cv2.circle(img,(cx,cy),5,(0,255,0),5,cv2.FILLED)
                    self.lmlist.append([id,cx,cy])
                    #print(cx,cy)
        return self.lmlist
    def findAngle(self, img, p1, p2, p3, draw=True):
 
        # Get the landmarks
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        x3, y3 = self.lmlist[p3][1:]
 
        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
 
        # print(angle)
 
        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle
def main():
    vid=cv2.VideoCapture(0)
    pt=0
    detector=posedetect()

    while True:
        success,img=vid.read()
        img=detector.findpose(img)
        lmlist=detector.findposition(img,draw=False)
        if len(lmlist)>14:
            print(lmlist[14])
            cv2.circle(img,(lmlist[14][1],lmlist[14][2]),5,(0,255,0),5,cv2.FILLED)
        ct=time.time()
        fps=1/(ct-pt)
        pt=ct
        cv2.putText(img,str(int(fps)), (40,120),cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 5,(255,0,0),4)
        cv2.imshow("Image",img)
    
        cv2.waitKey(1)

if __name__=="__main__":
    main()