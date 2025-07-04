## RUN main.py file

# Hand Gesture Controlled PDF Presentation

This project allows you to control a PDF presentation using hand gestures detected from a video file (or webcam). It extracts PDF pages as images and lets you navigate, annotate, and erase annotations using simple hand gestures.

## Features

- **PDF to Image Conversion:** Converts each page of a PDF into an image for display.
- **Hand Gesture Detection:** Uses computer vision to recognize hand gestures for navigation and annotation.
- **Video Input:** Processes gestures from a video file (can be adapted for webcam).
- **Slide Navigation:** Move to the next or previous slide using gestures.
- **Annotation:** Draw on slides using your index finger.
- **Erase:** Remove the last annotation with a gesture.
- **FPS Display:** Shows the current frames per second.

## Requirements

- Python 3.7+
- See `requirements.txt` for dependencies.

## Installation

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd HAND-GESTURE-2
   ```

2. **Create a virtual environment (optional but recommended):**
   ```sh
   python -m venv myenv
   myenv\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Place your PDF and video files:**
   - Put your PDF in the `documents` folder.
   - Put your video file (e.g., `movement.mp4`) in the project root or update the path in `video.py`.

## Usage

Run the main script:

```sh
python video.py
```

### Controls (Gestures)

- **Left Swipe (Thumb Up):** Go to the previous slide.
- **Right Swipe (Pinky Up):** Go to the next slide.
- **Pointer (Index & Middle Up):** Show pointer.
- **Draw (Index Up):** Draw on the slide.
- **Erase (Index, Middle, Ring Up):** Erase the last annotation.
- **Quit:** Press `q` on your keyboard.

## Project Structure

```
HAND-GESTURE-2/
│
├── documents/
│   ├── SIT_HACKAVERSE_2025_PPT_TEMPLATE[1].pdf
│   └── images/                # Extracted PDF pages as images
├── movement.mp4               # Video file for gesture detection
├── video.py                   # Main script
├── requirements.txt
└── .gitignore
```

## Dependencies

- `opencv-python`
- `numpy`
- `mediapipe`
- `matplotlib`
- `scikit-learn`
- `PyMuPDF` (fitz)
- `cvzone`

## Notes

- Make sure your video file and PDF paths are correct in `video.py`.
- The project can be adapted to use a webcam by changing the `cv2.VideoCapture` source.

---

**Author:** [Your Name]  
**License:** MIT (or your preferred license)