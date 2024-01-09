import mediapipe as mp
import cv2
import numpy as np
from flask import jsonify


class Pose_Estimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2)

    def estimate_pose(self, image):
        # Process the image with MediaPipe Pose.
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return 'No pose landmarks detected'

        # Extract pose landmarks.
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })

        return landmarks
