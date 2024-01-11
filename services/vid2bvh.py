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

    vid_landmarks = {}

    for index, frame in enumerate(frames):
        landmarks = pose_estimator.estimate_pose(frame)
        vid_landmarks[index] = landmarks

    return vid_landmarks


class vid2bvh:
    def __init__(self):
        self.pose_estimator = Pose_Estimator

    def convert(self, file):

        vid_landmarks = vid2landmarks(file)

        return ''
