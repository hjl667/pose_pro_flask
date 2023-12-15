from flask import Flask, request, jsonify
import mediapipe as mp
import cv2
import numpy as np

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

        # Process the image with MediaPipe Pose.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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

        return jsonify(landmarks)

    return 'Error processing file'


if __name__ == '__main__':
    app.run(debug=True)

