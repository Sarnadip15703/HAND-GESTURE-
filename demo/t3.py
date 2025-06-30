import fitz  # PyMuPDF
import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Initialize variables
width, height = 1280, 720
folderPath = "documents/images"  # Directory where images will be stored
pdfPath = "documents\SIT_HACKAVERSE_2025_PPT_TEMPLATE[1].pdf"  # Path to the PDF file

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Ensure the images directory exists
if not os.path.exists(folderPath):
    os.makedirs(folderPath)

# Extract PDF pages as images
def extract_pdf_pages(pdfPath, outputFolder):
    pdf = fitz.open(pdfPath)
    for pageNum in range(len(pdf)):
        page = pdf[pageNum]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Render at 2x resolution
        outputPath = os.path.join(outputFolder, f"page_{pageNum + 1}.png")
        pix.save(outputPath)
    pdf.close()

# Extract pages if not already extracted
if len(os.listdir(folderPath)) == 0:
    extract_pdf_pages(pdfPath, folderPath)

# Load images from the folder
images = sorted(os.listdir(folderPath), key=len)
imgNum = 0
hs, ws = int(120 * 1), int(190 * 1)
gestureThreshold = 500
buttonPressed = False
buttonCounter = 0
buttonDelay = 30
annotations = [[]]
annotationNumber = 0
annotationStart = False

detector = HandDetector(detectionCon=0.8, maxHands=1)

# Add variables to track hand movement
prev_cx = 0
movement_threshold = 20  # Minimum movement in pixels to detect a wave
neutral_zone = width // 3  # Define a neutral zone in the middle of the screen
direction = None  # Tracks the last detected direction ('left' or 'right')

while True:
    success, frame = cap.read()

    frame = cv2.flip(frame, 1)
    pathFullImage = os.path.join(folderPath, images[imgNum])
    imgCurr = cv2.imread(pathFullImage)

    # Check if the image was loaded successfully
    if imgCurr is None:
        print(f"Error: Unable to load image at {pathFullImage}")
        break

    imgCurr = cv2.resize(imgCurr, (1280, 740))

    hands, img = detector.findHands(frame)

    cv2.line(frame, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)

        # Use interpolated values for x and y
        lmlist = hand['lmList']  # List of landmarks
        xVal = int(np.interp(lmlist[20][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmlist[20][1], [150, height - 150], [0, height]))
        indexFinger = (xVal, yVal)

        # Update cx and cy with interpolated values
        cx, cy = xVal, yVal

        # Check if the hand is above the gesture threshold
        if cy <= gestureThreshold:
            # Detect waving gesture
            if abs(cx - prev_cx) > movement_threshold:
                # Gesture 1 - left
                if fingers == [0, 1, 1, 1, 1] and cx < prev_cx and direction != "left" and cx < (width - neutral_zone):  # Hand moved left
                    print("Wave Left")
                    if imgNum > 0:
                        buttonPressed = True
                        annotations = [[]]
                        annotationNumber = 0
                        annotationStart = False
                        imgNum -= 1
                    direction = "left"  # Lock the direction to left

                # Gesture 2 - right
                elif fingers == [0, 1, 1, 1, 1] and cx > prev_cx and direction != "right" and cx > (width - neutral_zone):  # Hand moved right
                    print("Wave Right")
                    if imgNum < len(images) - 1:
                        buttonPressed = True
                        annotations = [[]]
                        annotationNumber = 0
                        annotationStart = False
                        imgNum += 1
                    direction = "right"  # Lock the direction to right

        # Gesture 3 - show pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurr, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        # Gesture 4 - draw pointer
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart == False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            cv2.circle(imgCurr, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)
        else:
            annotationStart = False

        # Gesture 5 - erase
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                if annotationNumber >= 0:
                    annotations.pop()
                    annotationNumber -= 1
                    buttonPressed = True

        # Reset the direction lock only when the hand enters the neutral zone
        if neutral_zone < cx < (width - neutral_zone):
            direction = None

        # Update the previous center x-coordinate
        prev_cx = cx

    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonPressed = False
            buttonCounter = 0

    # Draw the annotations on the current image
    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv2.line(imgCurr, annotations[i][j - 1], annotations[i][j], (0, 0, 255), 12)

    imgSmall = cv2.resize(frame, (ws, hs))
    h, w, _ = imgCurr.shape

    imgCurr[0:hs, w - ws:w] = imgSmall

    cv2.imshow("Display picture", frame)
    cv2.imshow("Presentation", imgCurr)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()