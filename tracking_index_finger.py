import cv2
import mediapipe as mp
import time
import handTracking_module as htm
import numpy as np
import os
import re

prevX, prevY = 0, 0
pTime = 0
cTime = 0
savedTime = 0
cap = cv2.VideoCapture(0,cv2.CAP_MSMF)
detector = htm.handDetector(maxHand=1)
isStart = False



canvas = np.zeros((480, 640, 3), np.uint8)

if not os.path.exists("Strokes"):
    os.makedirs("Strokes")


def get_next_serial(save_dir, prefix="C"):
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.png$", re.IGNORECASE)
    used = set()

    for file_name in os.listdir(save_dir):
        match = pattern.match(file_name)
        if match:
            used.add(int(match.group(1)))

    next_serial = 0
    while next_serial in used:
        next_serial += 1

    return next_serial


next_count = get_next_serial("Strokes", "C")
while True:
    success , frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = detector.findHands(frame)
    lmList, bbox= detector.findPositions(frame)
    fingers = detector.fingersUp()
    h, w, c = frame.shape
    

    if len(lmList) != 0:
        x1, y1 = lmList[8][1], lmList[8][2]
        smooth_x =x1
        smooth_y =y1
        if fingers == [0,1,0,0,0]:
            smooth_x = int(prevX * 0.7 + smooth_x * 0.3)
            smooth_y = int(prevY * 0.7 + smooth_y * 0.3)
            cv2.line(canvas,(prevX,prevY),(smooth_x,smooth_y),(255,255,0),6)
        prevX , prevY = smooth_x ,smooth_y

        if fingers == [0,0,0,0,0]:
            print("finger closed to clear the canvas")
            canvas = np.zeros((480, 640, 3), np.uint8)

        if fingers[0] == 1 and sum(fingers[1:]) == 0:
            
            if time.time() - savedTime > 2:
                if np.sum(canvas) != 0:
                    cv2.imwrite(f"Strokes/D_{next_count}.png", canvas)
                    print(f"Saved: Strokes/C_{next_count}.png")
                    next_count += 1
                print("thumb up to saving image")
                savedTime = time.time()
           
        if fingers == [0,1,1,0,0]:
            distance = ((lmList[8][1] - lmList[12][1]) ** 2 + (lmList[8][2] - lmList[12][2]) ** 2) ** 0.5
            mX = (lmList[8][1] + lmList[12][1]) // 2
            mY = (lmList[8][2] + lmList[12][2]) // 2
            if distance < 35:
                cv2.circle(frame, (mX, mY), 25, (125, 200,255), cv2.FILLED)
                cv2.circle(canvas, (mX, mY), 25, (0,0,0), cv2.FILLED)
                print (distance)
        if fingers == [1,1,1,1,1]:
            print ("palm open to stop")

        
            
        # if(isStart):
        #     if prevX == 0 and prevY == 0:
        #         prevX, prevY = x1, y1
        #         smooth_x = int(prevX * 0.6 + x1 * 0.4)
        #         smooth_y = int(prevY * 0.6 + y1 * 0.4)
        #     cv2.line(canvas,(prevX,prevY),(smooth_x,smooth_y),(255,255,0),4)
        #     prevX , prevY = x1 , y1
 
    if not success:
        print("cannot read the vedio")
    # fps rate of images
    dTime = time.time() 
    fps = 1/(dTime - pTime)
    pTime = dTime

    cv2.putText(frame,f'FPS:{int(fps)}',(20,40),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),1)
    # Overlay canvas
    frame = cv2.addWeighted(frame, 1, canvas, 1, 0)
    cv2.imshow("handTracking",frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        print("quiting...")
        break    
cap.release()
cv2.destroyAllWindows()