import cv2
import mediapipe as mp
import time

vid=cv2.VideoCapture(0)
pt=0



mpDraw=mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose=mpPose.Pose()



while True:
    success,img=vid.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=pose.process(imgRGB)
    checker=(result.pose_landmarks)
    
    if checker:
        mpDraw.draw_landmarks(img,checker,mpPose.POSE_CONNECTIONS)
        for id,landmrk in enumerate(checker.landmark):
            #h,w,c=img.shape

            if id==0:
                h,w,c=img.shape
                cx,cy=(int(landmrk.x*w),int(landmrk.y*h))
                cv2.circle(img,(cx,cy),5,(0,255,0),5,cv2.FILLED)
                print(cx,cy)


    ct=time.time()
    fps=1/(ct-pt)
    pt=ct
    cv2.putText(img,str(int(fps)), (40,120),cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 5,(255,0,0),4)
    cv2.imshow("Image",img)
    
    cv2.waitKey(1)
