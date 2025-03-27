import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras

# Configuration
VIDEO_PATH = "/Users/marianareyes/Desktop/MatchMate/MatchMate/dataset/videos/BackhandP2.MOV"
#VIDEO_PATH = "/Users/marianareyes/Desktop/MatchMate/MatchMate/dataset/videos/ForehandP2.MOV"
#VIDEO_PATH = "/Users/marianareyes/Desktop/MatchMate/MatchMate/dataset/videos/ForehandK_9984.mov"
MODEL_PATH = "tennis_rnn.keras"
LEFT_HANDED = False
OUTPUT_FRAMES_PATH = "output_frames_labels"

# Initialize Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the trained model
model = keras.models.load_model(MODEL_PATH)

# Updated shot labels
shot_labels = ["follow_through", "bend_knees_finish_up", "bend_knees","racket_up", "follow_through", "perfect","neutral", "perfect"]

#shot_labels = ['forehand_follow_through', 'backhand_bend_knees_finish_up', 'forehand_bend_knees', 'forehand_racket_up', 'backhand_follow_through', 'backhand_perfect', 'neutral', 'forehand_perfect']
# Extract keypoints from Mediapipe
def extract_keypoints(results, left_handed=False):
    selected_landmarks = [
        'nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    keypoint_dict = {
        "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28
    }
    keypoints = []
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for name in selected_landmarks:
            idx = keypoint_dict[name]
            landmark = landmarks[idx]
            keypoints.extend([landmark.y, landmark.x])
        keypoints = np.array(keypoints)
        if left_handed:
            keypoints[1::2] = 1 - keypoints[1::2]
    else:
        keypoints = np.zeros(26)
    return keypoints

# Video processing
cap = cv2.VideoCapture(VIDEO_PATH)
features_pool = []
NB_IMAGES = 30
frame_id = 0
predicted_shot = 6  # default to neutral
probs = [0]*len(shot_labels)
probs[predicted_shot] = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    keypoints = extract_keypoints(results, LEFT_HANDED)
    features_pool.append(keypoints)

    if len(features_pool) == NB_IMAGES:
        features_seq = np.array(features_pool).reshape(1, NB_IMAGES, 26).astype(np.float32)
        probs = model.predict(features_seq, verbose=0)[0]
        predicted_shot = np.argmax(probs)
        features_pool.pop(0)

    # Draw pose
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Get bounding box
    if results.pose_landmarks:
        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark
        visible_landmarks = [lm for lm in landmarks if lm.visibility > 0.5]
        if visible_landmarks:
            x_min = min([lm.x for lm in visible_landmarks]) * w
            x_max = max([lm.x for lm in visible_landmarks]) * w
            y_min = min([lm.y for lm in visible_landmarks]) * h
            y_max = max([lm.y for lm in visible_landmarks]) * h

            if predicted_shot != 6:
                shot_text = shot_labels[predicted_shot]
                print(f"Feedback given: {shot_text}")
                cv2.putText(frame, shot_text.upper(), (int(x_min), int(y_min) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3, cv2.LINE_AA)

    # Draw vertical bar chart for probabilities
    bar_width = 20
    bar_spacing = 10
    bar_max_height = 100
    origin_x = frame.shape[1] - 40
    origin_y = 50 + bar_max_height

    for idx, prob in enumerate(probs):
        bar_height = int(prob * bar_max_height)
        top_left = (origin_x - idx * (bar_width + bar_spacing), origin_y - bar_height)
        bottom_right = (origin_x - idx * (bar_width + bar_spacing) + bar_width, origin_y)
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), -1)
        label = shot_labels[idx][:2].upper()  # short label
        cv2.putText(frame, label, (top_left[0], origin_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display frame
    cv2.imshow("Real-Time Tennis Shot Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pose.close()
cv2.destroyAllWindows()

print("DONE")