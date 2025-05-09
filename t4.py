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

while True:
    # Get the start time of the frame
    start_time = time.time()

    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    imgCurrent = cv2.resize(imgCurrent, (1280, 740))

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