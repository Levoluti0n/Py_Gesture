# 🤚 Hand Gesture Volume Control Application

This application utilizes **OpenCV** and **MediaPipe** to detect hand gestures and adjust the system volume based on the distance between the thumb and index finger. It captures video from your webcam and interprets hand movements in real-time.

---

## 🌟 Features

- Detects hand landmarks using MediaPipe.
- Adjusts system volume based on the distance between the thumb and index finger.
- Displays the current volume percentage and frame rate (FPS) on the video feed.

## 📦 Requirements

- Python 3.x
- OpenCV
- MediaPipe
- Additional system utility (`pactl`) for volume control (Linux)

## 🚀 Installation

1. **Install Python**: Ensure you have Python 3.x installed on your system.

2. **Install Required Libraries**: You can install the necessary libraries using pip:

   ```bash
   pip install opencv-python mediapipe
