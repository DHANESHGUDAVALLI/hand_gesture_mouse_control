import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import os
import logging
import tensorflow as tf
import math
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL, cast, POINTER

# Hide TensorFlow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.FATAL)

pyautogui.FAILSAFE = False

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, hand_no=0):
        lmList = []
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                handLms = self.results.multi_hand_landmarks[hand_no]
                h, w, _ = img.shape
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((id, cx, cy))
        return lmList

    def getHandType(self, hand_no=0):
        if self.results.multi_handedness:
            if hand_no < len(self.results.multi_handedness):
                return self.results.multi_handedness[hand_no].classification[0].label
        return None


# Volume Setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volumeControl = cast(interface, POINTER(IAudioEndpointVolume))
minVol, maxVol = volumeControl.GetVolumeRange()[:2]

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = HandDetector()
screen_width, screen_height = pyautogui.size()

prev_middle_y = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img = detector.findHands(img)

    if detector.results.multi_hand_landmarks:
        for hand_no in range(len(detector.results.multi_hand_landmarks)):
            handType = detector.getHandType(hand_no)
            lmList = detector.findPosition(img, hand_no)

            if lmList and len(lmList) >= 17:
                index_x, index_y = lmList[8][1], lmList[8][2]
                thumb_x, thumb_y = lmList[4][1], lmList[4][2]
                middle_x, middle_y = lmList[12][1], lmList[12][2]
                ring_x, ring_y = lmList[16][1], lmList[16][2]

                if handType == "Right":
                    # Mouse Movement
                    screen_x = np.interp(index_x, (100, 540), (0, screen_width))
                    screen_y = np.interp(index_y, (100, 380), (0, screen_height))
                    pyautogui.moveTo(screen_x, screen_y, duration=0.05)

                    # Left Click
                    if np.hypot(thumb_x - ring_x, thumb_y - ring_y) < 40:
                        pyautogui.click()

                    # Right Click
                    if np.hypot(thumb_x - middle_x, thumb_y - middle_y) < 40:
                        pyautogui.rightClick()

                    # Scrolling
                    if prev_middle_y != 0:
                        diff_y = middle_y - prev_middle_y
                        pyautogui.scroll(int(-diff_y * 2))
                    prev_middle_y = middle_y

                if handType == "Left":
                    # Volume Control
                    length = math.hypot(thumb_x - index_x, thumb_y - index_y)
                    vol = np.interp(length, [20, 200], [minVol, maxVol])
                    volumeControl.SetMasterVolumeLevel(vol, None)

                    cv2.putText(img, f'Volume: {int(length)}', (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Brightness Control
                    current_brightness = sbc.get_brightness(display=0)[0]

                    # Brightness Down
                    if middle_y > index_y and ring_y < index_y:
                        new_brightness = max(0, current_brightness - 2)
                        sbc.set_brightness(new_brightness)
                        cv2.putText(img, 'Brightness Down', (20, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Brightness Up
                    elif ring_y > index_y and middle_y < index_y:
                        new_brightness = min(100, current_brightness + 2)
                        sbc.set_brightness(new_brightness)
                        cv2.putText(img, 'Brightness Up', (20, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    else:
                        cv2.putText(img, 'Brightness Hold', (20, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Hand Gesture Mouse + Volume + Brightness Control", img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
