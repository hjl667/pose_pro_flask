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

    return m_landmarks_array


def vid2keypoints(img_path):
    img_set = []
    mediapipe_npy_path = '/Users/hongjiji/HJ/project/mediapipe_keypoints.npy'
    for img in os.listdir(img_path):
        if (img.endswith(".jpg")):
            keypoints = estimate(os.path.join(img_path, img))
            img_set.append(keypoints)

    np.save(mediapipe_npy_path, img_set)


def npy2bvh(npy_new_path):
    bvh_path = '/Users/hongjiji/HJ/project/exercise.bvh'
    my_instance = h36m_skeleton.H36mSkeleton()
    data = np.load(npy_new_path, allow_pickle=True)
    print(data.shape)
    my_instance.poses2bvh(data, None, bvh_path)
    print('')

def visualize_keypoints(keypoints_path):
    keypoint2index = {i: i for i in range(33)}  # Direct mapping for simplicity

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

    # Load keypoints data
    keypoints_data = np.load(keypoints_path)

    # Initialize plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()

    ax.set_xlim([keypoints_data[:, :, 0].min() - 0.1, keypoints_data[:, :, 0].max() + 0.1])
    ax.set_ylim([keypoints_data[:, :, 1].min() - 0.1, keypoints_data[:, :, 1].max() + 0.1])
    ax.set_zlim([keypoints_data[:, :, 2].min() - 0.1, keypoints_data[:, :, 2].max() + 0.1])

    ax.view_init(elev=12., azim=130)

    a = 165;
    X, Y, Z = keypoints_data[a].T
    # ax.scatter(X, Y, Z, s=50)

    for start_point, end_point in skeleton_connections:
        x_coords = [keypoints_data[a, start_point, 0], keypoints_data[a, end_point, 0]]
        y_coords = [keypoints_data[a, start_point, 1], keypoints_data[a, end_point, 1]]
        z_coords = [keypoints_data[a, start_point, 2], keypoints_data[a, end_point, 2]]
        ax.plot(x_coords, y_coords, z_coords, 'r')

    plt.show()


if __name__ == '__main__':
    vid_path = '/Users/hongjiji/HJ/project/exercise_clip.mp4'
    img_path = '/Users/hongjiji/HJ/project/sample_imgs'
    npy_path = '/Users/hongjiji/HJ/project/keypoints.npy'
    mediapipe_npy_path = '/Users/hongjiji/HJ/project/mediapipe_keypoints.npy'

    # video_to_frame(vid_path, img_path)
    # vid2keypoints(img_path)
    # reconstruct_npy(npy_path, npy_new_path)
    # npy2bvh(npy_new_path)
    visualize_keypoints(mediapipe_npy_path)