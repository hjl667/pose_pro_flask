import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)

# Load an image.
image_path = '/Users/hongjiji/HJ/project/sample_imgs/frame_165.jpg'
image = cv2.imread(image_path)

# Convert the image to RGB.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to get pose landmarks.
results = pose.process(image_rgb)

def plot_pose_landmarks_3d(landmarks, connections):
    fig = plt.figure(figsize=[15, 15])
    ax = fig.add_subplot(111, projection='3d')

    # Assuming the person is relatively flat on the plane, we need to scale the z-axis.
    custom_z_scale = 0.15  # This is a factor you might need to adjust manually

    # Extract normalized landmark coordinates.
    xs = [landmark[0] for landmark in landmarks]
    ys = [landmark[1] for landmark in landmarks]
    zs = [landmark[2] * custom_z_scale for landmark in landmarks]  # Apply the custom scale factor

    # Invert y-coordinates to match image coordinates system.
    ys = [1.0 - y for y in ys]

    # Plot the landmarks
    ax.scatter(xs, ys, zs, zdir='z', s=20, c='blue', depthshade=True)

    # Plot connections
    if connections:
        for connection in connections:
            start_point = landmarks.landmark[connection[0]]
            end_point = landmarks.landmark[connection[1]]
            ax.plot([start_point.x, end_point.x], [1.0 - start_point.y, 1.0 - end_point.y],
                    [start_point.z * custom_z_scale, end_point.z * custom_z_scale], 'r')

    # Set equal scaling
    ax.set_box_aspect([np.ptp(xs), np.ptp(ys), np.ptp(zs)])  # Aspect ratio is 1:1:1

    # Adjust viewing angle to match image perspective
    ax.view_init(elev=16., azim=-75)  # Adjust these angles to match your image's perspective

    plt.show()

def get_3d_keypoints_and_central_hip(image_path):
    image = cv2.imread(image_path)
    # Convert the BGR image to RGB before processing.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image.
    results = pose.process(image_rgb)

    keypoints = []

    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            # Keep the normalized coordinates.
            x, y, z = landmark.x, landmark.y, landmark.z
            keypoints.append((x, y, z))

        # Extract left and right hip coordinates in their normalized form.
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        # Calculate the central hip point in normalized coordinates.
        central_hip_x = (left_hip.x + right_hip.x) / 2
        central_hip_y = (left_hip.y + right_hip.y) / 2
        central_hip_z = (left_hip.z + right_hip.z) / 2

        # Add the central hip point to the keypoints list.
        central_hip_point = (central_hip_x, central_hip_y, central_hip_z)
        keypoints.append(central_hip_point)

        return keypoints
    else:
        print("No pose landmarks detected.")
        return None

# Replace 'image_path' with the path to your image file.
image_path = '/Users/hongjiji/HJ/project/sample_imgs/frame_165.jpg'
keypoints = get_3d_keypoints_and_central_hip(image_path)
plot_pose_landmarks_3d(keypoints, mp_pose.POSE_CONNECTIONS)


