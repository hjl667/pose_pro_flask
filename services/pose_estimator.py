import mediapipe as mp
import cv2
import numpy as np

def convert_mediapipe_to_h36m(mediapipe_pose):
    # Initialize the H36M pose with NaNs for each keypoint
    h36m_pose = {joint: np.array([np.nan, np.nan, np.nan]) for joint in range(17)}
    # Mapping from MediaPipe indices to H36M indices
    # Adjusted based on the order provided in the MediaPipe keypoints list
    mp_to_h36m_mapping = {
        23: 4,  # Left Hip
        25: 5,  # Left Knee
        27: 6,  # Left Ankle
        24: 1,  # Right Hip
        26: 2,  # Right Knee
        28: 3,  # Right Ankle
        11: 11,  # Left Shoulder
        13: 12,  # Left Elbow
        15: 13,  # Left Wrist
        12: 14,  # Right Shoulder
        14: 15,  # Right Elbow
        16: 16,  # Right Wrist
    }

    for mp_index, h36m_index in mp_to_h36m_mapping.items():
        h36m_pose[h36m_index] = mediapipe_pose[mp_index]

    h36m_pose[0] = (mediapipe_pose[23] + mediapipe_pose[24]) / 2  # Hip
    shoulders_mid = (mediapipe_pose[11] + mediapipe_pose[12]) / 2

    hips_mid = (mediapipe_pose[23] + mediapipe_pose[24]) / 2
    h36m_pose[7] = hips_mid + (shoulders_mid - hips_mid) * 0.5  # Spine
    h36m_pose[8] = shoulders_mid  # Thorax

    h36m_pose[9] = mediapipe_pose[0]  # Using Nose as Neck approximation
    eyes_mid = (mediapipe_pose[1] + mediapipe_pose[4]) / 2
    head_vector = eyes_mid - mediapipe_pose[0]
    h36m_pose[10] = mediapipe_pose[0] + head_vector * 2  # Rough estimation of HeadEndSite

    return h36m_pose

class Pose_Estimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2)

    def estimate_pose(self, image):
        # Process the image with MediaPipe Pose.
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return 'No pose landmarks detected'

        # Prepare landmarks for H36M conversion
        m_landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark]

        h36m_keypoints = convert_mediapipe_to_h36m(m_landmarks)

        return h36m_keypoints

if __name__ == '__main__':
    image_path = '../assets/sample_img.jpg'
    image = cv2.imread(image_path)
    pose_estimator = Pose_Estimator()
    landmarks = pose_estimator.estimate_pose(image)
    print(landmarks)
