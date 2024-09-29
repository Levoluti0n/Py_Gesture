import warnings

warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import time
import subprocess
import math


class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            mHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(mHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw and (
                    id == 4 or id == 8
                ):  # Thumb (id 4) and Index finger tip (id 8)
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList


def set_volume(volume):
    """Set system volume to a specific percentage using pactl."""
    volume = max(0, min(volume, 100))  # Clamp between 0 and 100
    subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{volume}%"])
    print(f"Volume set to {volume}%")


def map_distance_to_volume(dist, min_dist=50, max_dist=250, min_vol=0, max_vol=100):
    """Map the distance between thumb and index finger to a volume percentage."""
    # Clamp the distance between the defined minimum and maximum distances
    dist = max(min_dist, min(dist, max_dist))

    # Linearly map the distance to the volume range
    volume = ((dist - min_dist) / (max_dist - min_dist)) * (max_vol - min_vol) + min_vol
    return int(volume)


def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
            x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip

            # Draw a line between thumb and index finger
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 100:
                cv2.circle(
                    img,
                    ((x1 + x2) // 2, (y1 + y2) // 2),
                    15,
                    (0, 0, 255),
                    cv2.FILLED,
                )
            else:
                cv2.circle(
                    img, ((x1 + x2) // 2, (y1 + y2) // 2), 15, (255, 0, 255), cv2.FILLED
                )

            # Calculate the distance between the thumb and index finger

            # Map the distance to a volume percentage (0-100)
            volume = map_distance_to_volume(length)

            # Set the system volume based on the hand gesture distance
            set_volume(volume)

            # Display the volume percentage on the screen
            cv2.putText(
                img,
                f"Volume: {volume}%",
                (10, 50),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 255),
                3,
            )

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (10, 70),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 0, 255),
            3,
        )

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
