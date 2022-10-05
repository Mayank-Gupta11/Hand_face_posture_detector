import cv2
import mediapipe as mp
import time

vid=cv2.VideoCapture(0)

mphands=mp.solutions.hands
hands=mphands.Hands()
mpd=mp.solutions.drawing_utils
pt=0
ct=0

while True:
    success,img=vid.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #print(type(imgRGB))
    result = hands.process(imgRGB)
    chec=(result.multi_hand_landmarks)
    
    if (chec!=None):
        #print(chec)
        for hlms in chec:
            for id,lm in enumerate(hlms.landmark):
                #print(id,lm)
                h, w,c =img.shape
                cx, cy=int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                if id==4:
                    cv2.circle(img,(cx,cy),15,(225,230,0),cv2.FILLED)
            mpd.draw_landmarks(img,hlms,mphands.HAND_CONNECTIONS)
    cv2.flip(img,1,img)

    ct=time.time()
    fps=1/(ct-pt)
    pt=ct
    cv2.putText(img,str(int(fps)),(10,120),cv2.FONT_HERSHEY_DUPLEX,4,(235,190,0),5)

    cv2.imshow("Hand tracker",img)
    
    cv2.waitKey(1)

