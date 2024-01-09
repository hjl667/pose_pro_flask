from flask import Flask, request, jsonify
import mediapipe as mp
import cv2
import numpy as np
from services.pose_estimator import Pose_Estimator

app = Flask(__name__)

# Initialize MediaPipe Pose component.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file:
        # Read the image file in a format that can be processed.
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        pose_estimator = Pose_Estimator()
        results = jsonify(pose_estimator.estimate_pose(image))

        return results

    return 'Error processing file'


if __name__ == '__main__':
    app.run(debug=True)

