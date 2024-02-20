import cv2
import os
import mediapipe as mp
import numpy as np
import coco_skeleton
import os
import matplotlib.pyplot as plt

def estimate(img):
    img = cv2.imread(img)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        return 'No pose landmarks detected'

    # Prepare landmarks for H36M conversion
    m_landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark]
    m_landmarks_array = np.array(m_landmarks)

    m_landmarks_array = normalize_keypoints(m_landmarks_array)

    return m_landmarks_array

def normalize_keypoints(keypoints_data):
    # Calculate the minimum and maximum values for each dimension
    min_vals = keypoints_data.min(axis=0)
    max_vals = keypoints_data.max(axis=0)

    # Calculate the range of values for each dimension
    ranges = max_vals - min_vals

    # Normalize keypoints to range 0 to 1
    normalized_keypoints = (keypoints_data - min_vals) / ranges

    return normalized_keypoints

def visualize_keypoints(keypoints_data):
    # keypoint2index = {i: i for i in range(33)}  # Direct mapping for simplicity

    keypoints[:, [1, 2]] = keypoints[:, [2, 1]]

    skeleton_connections = [
        (11, 12),  # Left shoulder to Right shoulder
        (11, 13),  # Left shoulder to Left elbow
        (13, 15),  # Left elbow to Left wrist
        (12, 14),  # Right shoulder to Right elbow
        (14, 16),  # Right elbow to Right wrist
        (11, 23),  # Left shoulder to Left hip
        (12, 24),  # Right shoulder to Right hip
        (23, 24),  # Left hip to Right hip
        (23, 25),  # Left hip to Left knee
        (25, 27),  # Left knee to Left ankle
        (24, 26),  # Right hip to Right knee
        (26, 28),  # Right knee to Right ankle
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Calculate the center and range as before (fixed the center calculation to use 0.5 instead of 0.2)
    max_range = np.array([keypoints_data[:, 0].max() - keypoints_data[:, 0].min(),
                          keypoints_data[:, 1].max() - keypoints_data[:, 1].min(),
                          keypoints_data[:, 2].max() - keypoints_data[:, 2].min()]).max() / 2.0
    mid_x = (keypoints_data[:, 0].max() + keypoints_data[:, 0].min()) * 0.5
    mid_y = (keypoints_data[:, 1].max() + keypoints_data[:, 1].min()) * 0.5
    mid_z = (keypoints_data[:, 2].max() + keypoints_data[:, 2].min()) * 0.5

    # Set the limits for each axis to ensure equal aspect ratio
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y + max_range, mid_y - max_range)  # Invert the Y-axis
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set a viewing angle
    ax.view_init(elev=14., azim=20)

    # Plot the keypoints for the single frame
    X, Y, Z = keypoints_data.T
    # ax.scatter(X, Y, Z, s=50)

    # Draw lines for the skeleton for the single frame
    for start_point, end_point in skeleton_connections:
        x_coords = [keypoints_data[start_point, 0], keypoints_data[end_point, 0]]
        y_coords = [keypoints_data[start_point, 1], keypoints_data[end_point, 1]]
        z_coords = [keypoints_data[start_point, 2], keypoints_data[end_point, 2]]
        ax.plot(x_coords, y_coords, z_coords, 'r')

    for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        ax.text(x, y, z, '%d' % (i), size=5, zorder=1, color='k')

    plt.show()

def visualize_coco_keypoints(keypoints_data):
    keypoints_data[:, [1, 2]] = keypoints_data[:, [2, 1]]
    keypoints_data = normalize_keypoints(keypoints_data)
    skeleton_connections = [
        (5, 6),  # Left shoulder to Right shoulder
        (5, 7),  # Left shoulder to Left elbow
        (7, 9),  # Left elbow to Left wrist
        (6, 8),  # Right shoulder to Right elbow
        (8, 10),  # Right elbow to Right wrist
        (5, 11),  # Left shoulder to Left hip
        (6, 12),  # Right shoulder to Right hip
        (11, 12),  # Left hip to Right hip
        (11, 13),  # Left hip to Left knee
        (13, 15),  # Left knee to Left ankle
        (12, 14),  # Right hip to Right knee
        (14, 16),  # Right knee to Right ankle
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Calculate the center and range as before (fixed the center calculation to use 0.5 instead of 0.2)
    max_range = np.array([keypoints_data[:, 0].max() - keypoints_data[:, 0].min(),
                          keypoints_data[:, 1].max() - keypoints_data[:, 1].min(),
                          keypoints_data[:, 2].max() - keypoints_data[:, 2].min()]).max() / 2.0
    mid_x = (keypoints_data[:, 0].max() + keypoints_data[:, 0].min()) * 0.5
    mid_y = (keypoints_data[:, 1].max() + keypoints_data[:, 1].min()) * 0.5
    mid_z = (keypoints_data[:, 2].max() + keypoints_data[:, 2].min()) * 0.5

    # Set the limits for each axis to ensure equal aspect ratio
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y + max_range, mid_y - max_range)  # Invert the Y-axis
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set a viewing angle
    ax.view_init(elev=14., azim=20)

    # Plot the keypoints for the single frame
    X, Y, Z = keypoints_data.T
    # ax.scatter(X, Y, Z, s=50)

    # Draw lines for the skeleton for the single frame
    for start_point, end_point in skeleton_connections:
        x_coords = [keypoints_data[start_point, 0], keypoints_data[end_point, 0]]
        y_coords = [keypoints_data[start_point, 1], keypoints_data[end_point, 1]]
        z_coords = [keypoints_data[start_point, 2], keypoints_data[end_point, 2]]
        ax.plot(x_coords, y_coords, z_coords, 'r')

    for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        ax.text(x, y, z, '%d' % (i), size=5, zorder=1, color='k')

    plt.show()

def convert_mediapipe_to_coco(mediapipe_keypoints):
    coco_keypoints = np.zeros((17, 3))

    direct_mappings = {
        0: 0,  # Nose
        2: 1,  # LeftEye
        5: 2,  # RightEye
        7: 3,  # LeftEar
        8: 4,  # RightEar
        11: 5,  # LeftShoulder
        12: 6,  # RightShoulder
        13: 7,  # LeftElbow
        14: 8,  # RightElbow
        15: 9,  # LeftWrist
        16: 10,  # RightWrist
        23: 11,  # LeftHip
        24: 12,  # RightHip
        25: 13,  # LeftKnee
        25: 14,  # RightKnee
        27: 15,  # LeftAnkle
        28: 16,  # RightAnkle
    }

    for mp_index, coco_index in direct_mappings.items():
        coco_keypoints[coco_index] = mediapipe_keypoints[mp_index]
    print('ji')
    # # approximating neck [17]
    # coco_keypoints[17, :2] = (mediapipe_keypoints[11, :2] + mediapipe_keypoints[12, :2]) / 2
    # coco_keypoints[17, 2] = (mediapipe_keypoints[11, 2] + mediapipe_keypoints[12, 2]) / 2

    return coco_keypoints

if __name__ == '__main__':
    img_path = '/Users/hongjiji/HJ/project/sample_imgs/frame_162.jpg'
    # img_path = '/Users/hongjiji/HJ/project/yoga_pose.jpg'
    keypoints = estimate(img_path)
    coco_keypoints = convert_mediapipe_to_coco(keypoints)
    visualize_coco_keypoints(coco_keypoints)
    # visualize_keypoints(keypoints)

