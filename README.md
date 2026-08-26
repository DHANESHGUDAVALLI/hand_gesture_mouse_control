# 🖐️ Hand Gesture Mouse Control

> A touchless human-computer interaction system that uses real-time hand tracking to control mouse movement, clicking, scrolling, and system volume.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-00A98F)](https://ai.google.dev/edge/mediapipe/solutions/guide)

## 📌 Overview

**Hand Gesture Mouse Control** transforms hand movements into computer-control actions without requiring a physical mouse.

The system captures webcam frames, detects hand landmarks using MediaPipe, interprets landmark positions, and maps gestures to operating-system actions using PyAutoGUI and Windows audio-control libraries.

The current implementation supports:

- 🖱️ Cursor movement
- 👆 Left click
- 👉 Right click
- 🔄 Scrolling
- 🔊 Volume control
- ✋ Multi-hand tracking

## 🎯 Problem Statement

Traditional mouse interaction requires physical contact with an input device. Touchless interaction can be useful in accessibility, interactive demonstrations, public interfaces, presentations, and experimental human-computer interaction.

This project explores how computer vision and hand landmarks can create a natural gesture-based control layer.

## ✨ Key Features

### 🖱️ Mouse Movement

The index-finger position is mapped from camera coordinates to screen coordinates.

### 👆 Left Click

A thumb/ring-finger distance threshold is used to trigger a left click.

### 👉 Right Click

A thumb/middle-finger distance threshold is used to trigger a right click.

### 🔄 Scrolling

Changes in middle-finger vertical position are converted into scroll events.

### 🔊 Volume Control

The left hand is used to control system volume based on the distance between the thumb and index finger.

### ✋ Hand Tracking

The application detects and tracks hand landmarks and identifies handedness.

## 🧠 Interaction Pipeline

```text
Webcam
  ↓
OpenCV Frame Capture
  ↓
MediaPipe Hand Detection
  ↓
21 Hand Landmarks
  ↓
Gesture Interpretation
  ↓
PyAutoGUI / System Controls
  ↓
Computer Interaction
```

## 🏗️ Architecture

```text
┌──────────────┐
│    Webcam    │
└──────┬───────┘
       ↓
┌──────────────┐
│    OpenCV    │
└──────┬───────┘
       ↓
┌──────────────┐
│   MediaPipe  │
│ Hand Tracking│
└──────┬───────┘
       ↓
┌──────────────┐
│ Landmark &   │
│ Gesture Logic│
└──────┬───────┘
       ↓
┌──────────────┬───────────────┐
│   PyAutoGUI  │ System Audio  │
│ Mouse Control│   Control     │
└──────────────┴───────────────┘
```

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| Computer Vision | OpenCV |
| Hand Tracking | MediaPipe |
| Automation | PyAutoGUI |
| Numerical Processing | NumPy |
| Deep Learning Runtime | TensorFlow |
| Brightness | screen_brightness_control |
| Windows Audio | PyCaw |
| Windows COM | comtypes |

## 📂 Project Structure

```text
hand_gesture_mouse_control/
├── hand_gesture.py
├── code_2.py
├── firstcode.txt
├── exp1.txt
├── GIT_COMMANDS.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

> The current implementation uses Windows-specific system-control libraries for audio control.

```bash
git clone https://github.com/DHANESHGUDAVALLI/hand_gesture_mouse_control.git
cd hand_gesture_mouse_control

python -m venv venv
venv\Scripts\activate

pip install opencv-python mediapipe numpy pyautogui tensorflow screen-brightness-control pycaw comtypes
```

## ▶️ Run

Connect a webcam and run:

```bash
python hand_gesture.py
```

Grant camera access if requested.

## 🎮 Gesture Mapping

| Hand | Gesture / Movement | Action |
|---|---|---|
| Right | Index finger movement | Cursor movement |
| Right | Thumb + ring finger | Left click |
| Right | Thumb + middle finger | Right click |
| Right | Middle finger vertical movement | Scroll |
| Left | Thumb + index finger distance | Volume |

The exact gesture thresholds are implemented directly in the Python program and can be tuned for different cameras and environments.

## ⚠️ Practical Considerations

Gesture systems can be affected by:

- Lighting conditions
- Camera quality
- Hand occlusion
- Background clutter
- Distance from camera
- Detection latency
- Individual hand positioning

For better reliability, perform calibration and use stable gestures rather than rapid movements.

## 🌍 Real-World Applications

- ♿ Accessibility interfaces
- 🖥️ Touchless computer control
- 🎤 Presentation control
- 🏥 Touch-free environments
- 🧪 Human-computer interaction research
- 🎮 Gesture-based interfaces
- 🏭 Industrial touchless controls

## 🚀 Future Enhancements

- Custom gesture training
- Gesture configuration UI
- Cross-platform support
- Smoother cursor filtering
- Gesture cooldown/debouncing
- Application-specific gestures
- Brightness control gestures
- Media-playback controls
- Voice + gesture hybrid interaction
- ML-based gesture classification
- Calibration wizard

## 🔗 Repository

[View Hand Gesture Mouse Control on GitHub](https://github.com/DHANESHGUDAVALLI/hand_gesture_mouse_control)

## 👨‍💻 Author

**Dhanesh Gudavalli**

If this project is useful, consider giving the repository a ⭐.
