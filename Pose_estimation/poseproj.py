import pmodule as pm
import cv2
import mediapipe as mp
import time

vid=cv2.VideoCapture(0)
pt=0
detector=pm.posedetect()

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
