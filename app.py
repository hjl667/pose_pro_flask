from flask import Flask, request, jsonify, send_file
import mediapipe as mp
import cv2
import numpy as np
from services.pose_estimator import Pose_Estimator
from services.vid2bvh import vid2bvh

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


@app.route('/animate', method=['POST'])
def animate_video():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    if file:
        # logic to convert file to bvh
        bvhConverter = vid2bvh
        bvh_content = bvhConverter.convert(file)

        return send_file(bvh_content, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
