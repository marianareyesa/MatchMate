import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras

# Configuration
#VIDEO_PATH = "/Users/marianareyes/Desktop/MatchMate/MatchMate/dataset/videos/tennis_rally2.mov"
#VIDEO_PATH = "/Users/marianareyes/Desktop/MatchMate/MatchMate/dataset/videos/ForehandP_9974.MOV"
VIDEO_PATH = "/Users/marianareyes/Desktop/MatchMate/MatchMate/dataset/videos/BackhandP2.MOV"
#VIDEO_PATH = "/Users/marianareyes/Desktop/MatchMate/MatchMate/dataset/videos/ForehandR_9993.MOV"

MODEL_PATH = "tennis_rnn.keras"
MODEL_FEEDBACK_PATH = "tennis_rnn.keras"
LEFT_HANDED = False
OUTPUT_FRAMES_PATH = "output_frames"
OUTPUT_FRAMES_PATH_FEEDBACK = "output_frames_labels"

# Initialize Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the trained model
model = keras.models.load_model(MODEL_PATH)
model_feedback = keras.models.load_model(MODEL_FEEDBACK_PATH)

# ShotCounter class
class ShotCounter:
    MIN_FRAMES_BETWEEN_SHOTS = 60

    def __init__(self):
        self.nb_forehands = 0
        self.nb_backhands = 0
        self.nb_serves = 0
        self.last_shot = "neutral"
        self.frames_since_last_shot = self.MIN_FRAMES_BETWEEN_SHOTS

    def update(self, probs):
        if probs[0] > 0.80 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:
            self.nb_backhands += 1
            self.last_shot = "backhand"
            self.frames_since_last_shot = 0
        elif probs[1] > 0.80 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:
            self.nb_forehands += 1
            self.last_shot = "forehand"
            self.frames_since_last_shot = 0
        elif len(probs) > 3 and probs[3] > 0.80 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:
            self.nb_serves += 1
            self.last_shot = "serve"
            self.frames_since_last_shot = 0

        self.frames_since_last_shot += 1

    def display(self, frame):
        cv2.putText(frame, f"Forehands = {self.nb_backhands}", (20, frame.shape[0] - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if self.last_shot == "backhand" else (255, 0, 0), 2)
        cv2.putText(frame, f"Backhands = {self.nb_forehands}", (20, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if self.last_shot == "forehand" else (255, 0, 0), 2)
        cv2.putText(frame, f"Serves = {self.nb_serves}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if self.last_shot == "serve" else (255, 0, 0), 2)

# Extract keypoints from Mediapipe

def extract_keypoints(results, left_handed=False):
    selected_landmarks = [
        'nose',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle'
    ]
    
    keypoint_dict = {
        "nose": 0,
        "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
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
            keypoints.extend([landmark.y, landmark.x])  # explicitly y, x order as your model expects

        keypoints = np.array(keypoints)

        if left_handed:
            keypoints[1::2] = 1 - keypoints[1::2]  # invert x-coordinates for left-handed players
    else:
        keypoints = np.zeros(26)  # ensures correct shape even if landmarks missing

    return keypoints


# Video processing
cap = cv2.VideoCapture(VIDEO_PATH)

features_pool = []
NB_IMAGES = 30
shot_counter = ShotCounter()
frame_id = 0

# Initialize predicted_shot and probs outside the loop to prevent NameError
predicted_shot = 2  # default neutral
probs = [0, 0, 1, 0]  # default neutral (probability 100%)
shot_labels = ["forehand", "backhand", "neutral", "serve"]

shot_labels_feedback = ["follow_through", "bend_knees_finish_up", "bend_knees","racket_up", "follow_through", "perfect","neutral", "perfect"]
predicted_feedback = 6  # default to neutral
probs_feedback = [0]*len(shot_labels_feedback)
probs_feedback[predicted_feedback] = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    keypoints = extract_keypoints(results, LEFT_HANDED)

    features_pool.append(keypoints)

    if len(features_pool) == NB_IMAGES:
        features_seq = np.array(features_pool).reshape(1, NB_IMAGES, 26)
        features_seq_feedback = np.array(features_pool).reshape(1, NB_IMAGES, 26).astype(np.float32)

        probs = model.predict(features_seq, verbose=0)[0]
        probs_feedback = model_feedback.predict(features_seq_feedback, verbose=0)[0]

        shot_counter.update(probs)
        features_pool.pop(0)

        predicted_shot = np.argmax(probs)
        predicted_feedback = np.argmax(probs_feedback)

    # Draw landmarks
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Get bounding box coordinates
    if results.pose_landmarks:
        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark
        visible_landmarks = [lm for lm in landmarks if lm.visibility > 0.5]
        if visible_landmarks:  # ensure you have visible landmarks
            x_min = min([lm.x for lm in visible_landmarks]) * w
            x_max = max([lm.x for lm in visible_landmarks]) * w
            y_min = min([lm.y for lm in visible_landmarks]) * h
            y_max = max([lm.y for lm in visible_landmarks]) * h

            if predicted_feedback != 6:
                shot_text = shot_labels_feedback[predicted_feedback]
                print(f"Feedback given: {shot_text}")
                cv2.putText(frame, shot_text.upper(), (int(x_min), int(y_min) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)

            # Draw bounding box
            #cv2.rectangle(frame, (int(x_min)-20, int(y_min)-20), (int(x_max)+20, int(y_max)+20),
                          #(0, 255, 255), 3)

            # Draw shot label
            if predicted_shot != 2:  # ignore neutral
                shot_text = shot_labels[predicted_shot]
                #cv2.putText(frame, shot_text.upper(), (int(x_min), int(y_min) - 30),
                            #cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3, cv2.LINE_AA)

    # Display shot counters
    shot_counter.display(frame)

    # Display shot probability bars
    # Draw vertical shot probability bars (histogram style)
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

        # Label (F, B, N, S) underneath the bar
        cv2.putText(frame, shot_labels[idx][0].upper(),
                    (top_left[0] + 3, origin_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Real-time display window
    cv2.imshow("Real-Time Tennis Shot Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pose.close()
cv2.destroyAllWindows()

print("Final counts:")
print(f"Forehands: {shot_counter.nb_backhands}")
print(f"Backhands: {shot_counter.nb_forehands}")
print(f"Serves: {shot_counter.nb_serves}")
