import cv2
import mediapipe as mp
import time



class handDetector():
    def __init__(self,mode = False,maxHand = 2, detectionCon = 0.5, trackCon = 0.5, modelComplexity = 0):
        self.mode = mode
        self.maxHand = maxHand
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComplexity = modelComplexity

        # mediapipe module 
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHand,
            model_complexity=self.modelComplexity,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self,frame,draw = True):
        imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks) # detect the handlandmarks coordinates 
        if self.result and self.result.multi_hand_landmarks :
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame,handLms,self.mpHands.HAND_CONNECTIONS)

        return frame
    
    def findPositions(self,frame, HandNo = 0 , draw = True):
        self.lmlist = []
        # for boundary box ko lai
        xlist = []
        ylist = []
        bbox = None

        
        if self.result.multi_hand_landmarks :
            myHand = self.result.multi_hand_landmarks[HandNo]
            for id, lm in enumerate(myHand.landmark): #for each hand handlandmark id and lm

                # print(id,lm)
                h, w, c  = frame.shape 
                CX, CY = int(lm.x*w), int(lm.y*h)  
                self.lmlist.append([id,CX,CY])  
                xlist.append(CX)
                ylist.append(CY)         
                if draw:
                    if id == 8  or id == 4 :
                        cv2.circle(frame,(CX,CY),8,(0,255,0),cv2.FILLED)
            if len(xlist) != 0 and len(ylist) != 0:
                Xmin, Xmax = min(xlist), max(xlist)              
                Ymin, Ymax = min(ylist), max(ylist) 
                bbox = (Xmin, Ymin, Xmax, Ymax) 
                if draw:
                    cv2.rectangle(frame,(Xmin-20,Ymin-20),(Xmax+20,Ymax+20),(255,0,0),2)    
                       
        return self.lmlist, bbox
    
    # finger up/down check
    def fingersUp(self):
        fingers = []
        
        if len(self.lmlist) < 5 :
            return [0,0,0,0,0]

        # x coordinate
        if self.lmlist[4][1] < self.lmlist[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # y coordinate
        tipId = [8,12,16,20]
        for id in tipId:
            if self.lmlist[id][2] < self.lmlist[id - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers

        
   


