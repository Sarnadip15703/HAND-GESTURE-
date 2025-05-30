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
gestureThreshold = 500
buttonPressed = False
buttonCounter = 0
buttonDelay = 15

detector = HandDetector(detectionCon=0.8, maxHands=1)

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

        xVal = int(np.interp(lmlist[8][0], [width//2, ws], [0, width]))
        yVal = int(np.interp(lmlist[8][1], [150, height-150], [0, height]))
        indexFinger = (xVal, yVal)
        
        if cy <= gestureThreshold: #If hand is at height of face
            # gesture 1 - left
            if fingers == [0, 1, 0, 0, 0]:
                print("Left", imgNum)
                if imgNum > 0:
                    buttonPressed = True
                    imgNum -= 1

            #gesture 2 - right
            if fingers == [0, 0, 0, 0, 1] and lmlist[20][0] > lmlist[19][0]:
                print("right", imgNum)
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






"""

FOR ZOOM IN AND OUT


from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np
import time

# Parameters
width, height = 1280, 720
gestureThreshold = 500
folderPath = "Presentation"

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Paths
folderPath = "documents/images"  # Directory where images will be stored
pdfPath = "documents/SIT_HACKAVERSE_2025_PPT_TEMPLATE[1].pdf"  # Path to the PDF file

if not os.path.exists(folderPath):
    os.makedirs(folderPath)

# Extract PDF pages as images
def extract_pdf_pages(pdfPath, outputFolder):
    import fitz  # PyMuPDF
    pdf = fitz.open(pdfPath)
    for pageNum in range(len(pdf)):
        page = pdf[pageNum]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Render at 2x resolution
        outputPath = os.path.join(outputFolder, f"page_{pageNum + 1}.png")
        pix.save(outputPath)
    pdf.close()

if len(os.listdir(folderPath)) == 0:
    extract_pdf_pages(pdfPath, folderPath)

# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# Variables
imgList = []
delay = 30
buttonPressed = False
counter = 0
drawMode = False
imgNumber = 0
delayCounter = 0
annotations = [[]]
annotationNumber = -1
annotationStart = False
hs, ws = int(120 * 1), int(213 * 1)  # width and height of small image

# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

# Set the target frame rate
target_fps = 30
frame_duration = 1 / target_fps  # Duration of each frame in seconds

# Initialize variables for FPS calculation and button delay
prevTime = time.time()
buttonPressed = False
buttonDelay = 30  # Delay in frames to prevent rapid transitions
buttonCounter = 0

# Zoom variables
zoom_level = 1.0
zoom_min = 1.0
zoom_max = 2.0
zoom_step = 0.1

while True:
    # Get the start time of the frame
    start_time = time.time()

    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    imgCurrent = cv2.resize(imgCurrent, (1280, 740))

    # --- ZOOM LOGIC ---
    zoomed_img = imgCurrent.copy()
    if zoom_level != 1.0:
        h, w = imgCurrent.shape[:2]
        new_w, new_h = int(w * zoom_level), int(h * zoom_level)
        zoomed_img = cv2.resize(imgCurrent, (new_w, new_h))

        # Crop or pad to keep the image centered and at original size
        if zoom_level > 1.0:
            # Crop center
            x1 = (new_w - w) // 2
            y1 = (new_h - h) // 2
            zoomed_img = zoomed_img[y1:y1 + h, x1:x1 + w]
        else:
            # Pad center
            pad_w = (w - new_w) // 2
            pad_h = (h - new_h) // 2
            zoomed_img = cv2.copyMakeBorder(
                zoomed_img, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
    imgCurrent = zoomed_img
    # --- END ZOOM LOGIC ---

    # Find the hand and its landmarks
    hands, img = detectorHand.findHands(img)  # with draw
    # Draw Gesture Threshold line
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and buttonPressed is False:  # If hand is detected
        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"]  # List of 21 Landmark points
        fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

        # Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height-150], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:  # If hand is at the height of the face
            if fingers == [1, 0, 0, 0, 0]:
                print("Left")
                buttonPressed = True
                if imgNumber > 0:
                    imgNumber -= 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
            if fingers == [0, 0, 0, 0, 1]:
                print("Right")
                buttonPressed = True
                if imgNumber < len(pathImages) - 1:
                    imgNumber += 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False

        # Zoom In: Thumb and index up
        if fingers == [1, 1, 0, 0, 0]:
            zoom_level = min(zoom_level + zoom_step, zoom_max)
            print(f"Zoom In: {zoom_level:.1f}x")
            buttonPressed = True

        # Zoom Out: Index and middle up
        if fingers == [1, 1, 1, 0, 0]:
            zoom_level = max(zoom_level - zoom_step, zoom_min)
            print(f"Zoom Out: {zoom_level:.1f}x")
            buttonPressed = True

        if fingers == [0, 1, 0, 0, 1]:
            time.sleep(1)
            break

        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            print(annotationNumber)
            annotations[annotationNumber].append(indexFinger)
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        else:
            annotationStart = False

        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

    else:
        annotationStart = False

    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    for i, annotation in enumerate(annotations):
        for j in range(len(annotation)):
            if j != 0:
                cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)

    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws: w] = imgSmall

    # Calculate FPS
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    # Display FPS on the screen
    cv2.putText(img, f"FPS: {int(fps)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Slides", imgCurrent)
    cv2.imshow("Image", img)

    # Calculate the time taken to process the frame
    elapsed_time = time.time() - start_time

    # Add a delay to maintain the target frame rate
    wait_time = max(1, int((frame_duration - elapsed_time) * 1000))
    key = cv2.waitKey(wait_time)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""