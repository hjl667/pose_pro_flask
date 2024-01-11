import mediapipe as mp
import cv2

def convert_mediapipe_to_h36m(mediapipe_keypoints):
    mediapipe_keypoints_labels = []
    h36m_keypoints_labels = []

    mapping = {
        'Nose': 'Head',
        'Left Eye': None,
        'Right Eye': None,
        'Left Ear': None,
        'Right Ear': None,
        'Mouth Left': None,
        'Mouth Right': None,
        'Left Shoulder': 'Left Arm',
        'Right Shoulder': 'Right Arm',
        'Left Elbow': 'Left Forearm',
        'Right Elbow': 'Right Forearm',
        'Left Wrist': 'Left Hand',
        'Right Wrist': 'Right Hand',
        'Left Pinky': None,
        'Right Pinky': None,
        'Left Index': None,
        'Right Index': None,
        'Left Thumb': None,
        'Right Thumb': None,
        'Left Hip': 'Left Up Leg',
        'Right Hip': 'Right Up Leg',
        'Left Knee': 'Left Leg',
        'Right Knee': 'Right Leg',
        'Left Ankle': 'Left Foot',
        'Right Ankle': 'Right Foot',
        'Left Heel': None,
        'Right Heel': None,
        'Left Foot Index': None,
        'Right Foot Index': None
    }

    for mp_label, h36m_label in mapping.items():
        if h36m_label is not None:
            mediapipe_keypoints_labels.append(mp_label)
            h36m_keypoints_labels.append(h36m_label)

    for i in mediapipe_keypoints:
        if human36m_keypoints[i] is not None:

            human36m_keypoints = {
                'x': mediapipe_keypoints[i]['x'],
                'y': mediapipe_keypoints[i]['y'],
                'z': mediapipe_keypoints[i]['z'],
                'label': human36m_keypoints[i]
            }

    return human36m_keypoints


class Pose_Estimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2)

    def estimate_pose(self, image):
        # Process the image with MediaPipe Pose.
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return 'No pose landmarks detected'

        landmarks = []

        for landmark in results.pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
            })

        h36m_keypoints, = convert_mediapipe_to_h36m(landmarks)

        return h36m_keypoints


        # Extract pose landmarks.
        # landmarks labels from mediapipe:
        # labels = {
        #     "Nose", "Left Eye Inner", "Left Eye", "Left Eye Outer",
        #     "Right Eye Inner", "Right Eye", "Right Eye Outer",
        #     "Left Ear", "Right Ear", "Mouth Left", "Mouth Right",
        #     "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
        #     "Left Wrist", "Right Wrist", "Left Pinky", "Right Pinky",
        #     "Left Index", "Right Index", "Left Thumb", "Right Thumb",
        #     "Left Hip", "Right Hip", "Left Knee", "Right Knee",
        #     "Left Ankle", "Right Ankle", "Left Heel", "Right Heel",
        #     "Left Foot Index", "Right Foot Index"
        # };

if __name__ == '__main__':
    image_path = '../assets/sample_img.jpg'
    image = cv2.imread(image_path)
    pose_estimator = Pose_Estimator()
    landmarks = pose_estimator.estimate_pose(image)
    print(landmarks)
