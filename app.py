import cv2
import math
import numpy as np
import time
import streamlit as st
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import handtrackingModule as htm  # Ensure this is in the same folder as your app or install correctly

# Set up the Streamlit app
st.title("Volume Control Using Hand Gestures")

# Streamlit sidebar controls
st.sidebar.header("Settings")
detection_confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.7)
frame_width = st.sidebar.slider("Frame Width", 320, 1280, 640, step=10)
frame_height = st.sidebar.slider("Frame Height", 240, 720, 480, step=10)

# Initialize components
cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

detector = htm.handDetector(detectionCon=detection_confidence)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
)
volume = interface.QueryInterface(IAudioEndpointVolume)
vol_range = volume.GetVolumeRange()
min_volume = vol_range[0]
max_volume = vol_range[1]

volper = 0
volbar = 400
ptime = 0

# Streamlit placeholder for video frame
frame_placeholder = st.empty()

# Run the app and capture frames
while cap.isOpened():
    success, img = cap.read()
    if not success:
        st.error("Failed to capture video. Please check your camera.")
        break

    # Process the frame with hand tracking
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)

    if len(lmlist) != 0:
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

        # Calculate the distance between fingers
        length = math.hypot(x2 - x1, y2 - y1)

        # Map the distance to volume control
        vol = np.interp(length, [50, 300], [min_volume, max_volume])
        volbar = np.interp(length, [50, 300], [400, 150])
        volper = np.interp(length, [50, 300], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    # Draw the volume bar
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volbar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volper)} %', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Calculate FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # Streamlit image display
    frame_placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

    # Break loop on Streamlit button click
    if st.sidebar.button("Stop"):
        break

cap.release()
cv2.destroyAllWindows()
