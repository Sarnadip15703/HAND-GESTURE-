import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np

width, height = 1280, 720
folderPath = "Presentation"

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

images = sorted(os.listdir(folderPath), key=len)
imgNum = 0
hs, ws = int(150*1), int(250*1)
gestureThreshold = 300
buttonPressed = False
buttonCounter = 0
buttonDelay = 30

detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, frame = cap.read()

    frame = cv2.flip(frame, 1)
    pathFullImage = os.path.join(folderPath, images[imgNum])
    imgCurr = cv2.imread(pathFullImage)

    hands, img =  detector.findHands(frame)
    cv2.line(frame, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)


    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']

        lmlist = hand['lmList']

        xVal = int(np.interp(lmlist[8][0], [width//2, w], [0, width]))
        yVal = int(np.interp(lmlist[8][1], [150, height-150], [0, height]))
        indexFinger = (xVal, yVal)
        
        if cy <= gestureThreshold: #If hand is at height of face
            # gesture 1 - left
            if fingers == [1, 0, 0, 0, 0]:
                print("Left")
                if imgNum > 0:
                    buttonPressed = True
                    imgNum -= 1

            #gesture 2 - right
            if fingers == [0, 0, 0, 0, 1]:
                print("right")
                if imgNum < len(images)-1:
                    buttonPressed = True
                    imgNum += 1
                

        # Gesture 3 -show pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurr, indexFinger, 12, (0, 0, 255), cv2.FILLED)

    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonPressed = False
            buttonCounter = 0


    imgSmall = cv2.resize(frame, (ws, hs))
    h, w, _ = imgCurr.shape

    imgCurr[0:hs, w-ws:w] = imgSmall

    cv2.imshow("Display picture", frame)
    cv2.imshow("Presentation", imgCurr)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break