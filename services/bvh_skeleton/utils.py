import cv2
import os
import mediapipe as mp
import numpy as np
import h36m_skeleton
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def convert_mediapipe_to_h36m(mediapipe_keypoints):
    h36m_keypoints = np.zeros((17, 3))

    direct_mappings = {
        23: 0,  # Hip (average of left and right hips, will adjust later)
        24: 1,  # RightHip
        26: 2,  # RightKnee
        28: 3,  # RightAnkle
        25: 5,  # LeftKnee
        27: 6,  # LeftAnkle
        11: 11,  # LeftShoulder
        13: 12,  # LeftElbow
        15: 13,  # LeftWrist
        12: 14,  # RightShoulder
        14: 15,  # RightElbow
        16: 16,  # RightWrist
    }

    # Fill the directly mapped keypoints
    for mp_index, h36m_index in direct_mappings.items():
        h36m_keypoints[h36m_index] = mediapipe_keypoints[mp_index]

    h36m_keypoints[0] = (mediapipe_keypoints[23] + mediapipe_keypoints[24]) / 2

    shoulders_midpoint = (mediapipe_keypoints[11] + mediapipe_keypoints[12]) / 2
    h36m_keypoints[7] = (h36m_keypoints[0] + shoulders_midpoint) / 2

    neck_approx = shoulders_midpoint + (mediapipe_keypoints[0] - shoulders_midpoint) * 0.1
    h36m_keypoints[9] = neck_approx

    h36m_keypoints[8] = (h36m_keypoints[7] + neck_approx) / 2

    nose_to_neck_vector = neck_approx - mediapipe_keypoints[0]
    head_end_site_approx = mediapipe_keypoints[0] + 1.5 * nose_to_neck_vector
    h36m_keypoints[10] = head_end_site_approx

    return h36m_keypoints


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

    h36m_keypoints = convert_mediapipe_to_h36m(m_landmarks_array)

    return h36m_keypoints


def vid2keypoints(img_path):
    img_set = []
    save_path = '/Users/hongjiji/HJ/project/keypoints.npy'
    for img in os.listdir(img_path):
        if (img.endswith(".jpg")):
            keypoints = estimate(os.path.join(img_path, img))
            img_set.append(keypoints)

    np.save(save_path, img_set)

def npy2bvh(npy_new_path):
    bvh_path = '/Users/hongjiji/HJ/project/exercise.bvh'
    my_instance = h36m_skeleton.H36mSkeleton()
    data = np.load(npy_new_path, allow_pickle=True)
    print(data.shape)
    my_instance.poses2bvh(data, None, bvh_path)
    print('')


def visualize_keypoints(keypoints_path):
    keypoint2index = {
        'Hip': 0,
        'RightHip': 1,
        'RightKnee': 2,
        'RightAnkle': 3,
        'LeftHip': 4,
        'LeftKnee': 5,
        'LeftAnkle': 6,
        'Spine': 7,
        'Thorax': 8,
        'Neck': 9,
        'HeadEndSite': 10,
        'LeftShoulder': 11,
        'LeftElbow': 12,
        'LeftWrist': 13,
        'RightShoulder': 14,
        'RightElbow': 15,
        'RightWrist': 16,
    }

    skeleton_connections = [
        ('Hip', 'RightHip'),
        ('RightHip', 'RightKnee'),
        ('RightKnee', 'RightAnkle'),
        ('Hip', 'LeftHip'),
        ('LeftHip', 'LeftKnee'),
        ('LeftKnee', 'LeftAnkle'),
        ('Hip', 'Spine'),
        ('Spine', 'Thorax'),
        ('Thorax', 'Neck'),
        ('Neck', 'HeadEndSite'),
        ('Thorax', 'LeftShoulder'),
        ('LeftShoulder', 'LeftElbow'),
        ('LeftElbow', 'LeftWrist'),
        ('Thorax', 'RightShoulder'),
        ('RightShoulder', 'RightElbow'),
        ('RightElbow', 'RightWrist'),
    ]

    # Load keypoints data
    keypoints_data = np.load(keypoints_path)

    # Initialize plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set the limits of the plot based on keypoints data ranges
    ax.set_xlim([keypoints_data[:, :, 0].min() - 0.1, keypoints_data[:, :, 0].max() + 0.1])
    ax.set_ylim([keypoints_data[:, :, 1].min() - 0.1, keypoints_data[:, :, 1].max() + 0.1])
    ax.set_zlim([keypoints_data[:, :, 2].min() - 0.1, keypoints_data[:, :, 2].max() + 0.1])

    # Set consistent viewing angles
    ax.view_init(elev=10., azim=120)

    # Plot keypoints for the first frame
    X, Y, Z = keypoints_data[0].T
    ax.scatter(X, Y, Z, s=50)

    # Draw lines for the skeleton for the first frame
    for connection in skeleton_connections:
        start_point = keypoint2index[connection[0]]
        end_point = keypoint2index[connection[1]]
        x_coords = [keypoints_data[0, start_point, 0], keypoints_data[0, end_point, 0]]
        y_coords = [keypoints_data[0, start_point, 1], keypoints_data[0, end_point, 1]]
        z_coords = [keypoints_data[0, start_point, 2], keypoints_data[0, end_point, 2]]
        ax.plot(x_coords, y_coords, z_coords, 'r')

    # Show plot
    plt.show()


if __name__ == '__main__':
    vid_path = '/Users/hongjiji/HJ/project/exercise_clip.mp4'
    img_path = '/Users/hongjiji/HJ/project/sample_imgs'
    npy_path = '/Users/hongjiji/HJ/project/keypoints.npy'

    # video_to_frame(vid_path, img_path)
    # vid2keypoints(img_path)
    # reconstruct_npy(npy_path, npy_new_path)
    npy2bvh(npy_path)
    # visualize_keypoints(npy_path)
