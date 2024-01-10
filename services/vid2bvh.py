from io import BytesIO

import cv2

from pose_estimator import Pose_Estimator
def vid2landmarks(file):
    pose_estimator = Pose_Estimator

    video_bytes = BytesIO(file.read())
    video_bytes.seek(0)

    cap = cv2.VideoCapture(video_bytes)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    for frame in frames:
        landmarks = ''

    return landmarks

class vid2bvh:
    def __init__(self):
        self.count = 1

    def convert(self, file):
        return ''