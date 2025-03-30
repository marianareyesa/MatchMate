import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from speech_utils import give_feedback
from datetime import datetime
import random

# Configuration
VIDEO_PATH = "/Users/marianareyes/Desktop/MatchMate/MatchMate/dataset/IMG_0450.MOV"
MODEL_PATH = "models/tennis_rnn.keras"
MODEL_FEEDBACK_PATH = "tennis_rnn_feedback_new2.keras"
LEFT_HANDED = False
OUTPUT_FRAMES_PATH = "output_frames"

# Initialize Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the trained models
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

def normalize_keypoints(sequence):
    x = sequence[:, 0::2]
    y = sequence[:, 1::2]
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    x = (x - x_min) / (x_max - x_min + 1e-6)
    y = 1 - ((y - y_min) / (y_max - y_min + 1e-6))
    sequence[:, 0::2] = x
    sequence[:, 1::2] = y
    return sequence

# Labels
shot_labels = ["forehand", "backhand", "neutral", "serve"]
detailed_labels = [
    'backhand_follow_through', 'forehand_finish_up', 'neutral',
    'forehand_bend_knees', 'forehand_racket_up', 'forehand_follow_through',
    'backhand_perfect', 'backhand_bend_knees_finish_up',
    'backhand_finish_up', 'backhand_racket_up', 'forehand_perfect'
]

label_groups = {
    "perfect": ["forehand_perfect", "backhand_perfect"],
    "follow_through": ["forehand_follow_through", "backhand_follow_through"],
    "bend_knees": ["forehand_bend_knees", "backhand_bend_knees_finish_up"],
    "racket_up": ["forehand_racket_up", "backhand_racket_up"],
    "finish_up": ["forehand_finish_up", "backhand_finish_up"],
    "neutral": ["neutral"]
}

shot_labels_human_feedback = {
    "follow_through": ["Nice job, but remember to finish your swing.", "Complete that follow through for more control.", "Try not to stop the motion early—follow through fully."],
    "bend_knees": ["Lower your stance—bend your knees more.", "Keep your knees bent to stay balanced.", "Try bending your knees to generate more power."],
    "racket_up": ["Prepare early—get your racket up.", "Raise your racket sooner to be ready.", "Get your racket into position for the next shot."],
    "perfect": ["That was perfect—keep it up!", "Excellent form!", "Beautiful shot!"],
    "finish_up": ["Finish up!", "Finish with your racket all the way up"],
    "neutral": [""]
}

# Main loop
cap = cv2.VideoCapture(VIDEO_PATH)
features_pool = []
NB_IMAGES = 30
frame_id = 0
predicted_shot = 2
predicted_feedback = 2
probs_shot = [0, 0, 1, 0]
probs_feedback = np.zeros(len(detailed_labels))
shot_counter = ShotCounter()
feedback_stats = {label: 0 for label in shot_labels_human_feedback.keys()}

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
        probs_shot = model.predict(features_seq, verbose=0)[0]
        shot_counter.update(probs_shot)
        predicted_shot = np.argmax(probs_shot)

        normalized = normalize_keypoints(np.array(features_pool).copy())
        input_seq = normalized.reshape(1, NB_IMAGES, 26).astype(np.float32)
        probs_feedback = model_feedback.predict(input_seq, verbose=0)[0]
        predicted_feedback = np.argmax(probs_feedback)
        features_pool.pop(0)

    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    grouped_probs = {group: 0.0 for group in label_groups}
    for group, members in label_groups.items():
        for m in members:
            if m in detailed_labels:
                idx = detailed_labels.index(m)
                grouped_probs[group] += probs_feedback[idx]

    top_feedback = max(grouped_probs.items(), key=lambda x: x[1])

    if results.pose_landmarks and top_feedback[0] != "neutral":
        h, w, _ = frame.shape
        visible_landmarks = [lm for lm in results.pose_landmarks.landmark if lm.visibility > 0.5]
        if visible_landmarks:
            x_min = int(min([lm.x for lm in visible_landmarks]) * w)
            y_min = int(min([lm.y for lm in visible_landmarks]) * h)
            cv2.putText(frame, f"{top_feedback[0].upper()} ({top_feedback[1]*100:.1f}%)", (x_min, y_min - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            feedback_options = shot_labels_human_feedback.get(top_feedback[0], [])
            if feedback_options and predicted_shot != 2:
                feedback_stats[top_feedback[0]] += 1
                give_feedback(random.choice(feedback_options))

    shot_counter.display(frame)

    bar_width, bar_spacing, bar_max_height = 20, 10, 100
    origin_x = frame.shape[1] - 40
    origin_y = 50 + bar_max_height

    for idx, prob in enumerate(probs_shot):
        bar_height = int(prob * bar_max_height)
        top_left = (origin_x - idx * (bar_width + bar_spacing), origin_y - bar_height)
        bottom_right = (origin_x - idx * (bar_width + bar_spacing) + bar_width, origin_y)
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), -1)
        label = shot_labels[idx][0].upper()
        cv2.putText(frame, label, (top_left[0], origin_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("Real-Time Tennis Shot Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pose.close()
cv2.destroyAllWindows()

print("\nFinal counts:")
print(f"Forehands: {shot_counter.nb_backhands}")
print(f"Backhands: {shot_counter.nb_forehands}")
print(f"Serves: {shot_counter.nb_serves}")

print("\nFeedback Summary:")
for label, count in feedback_stats.items():
    if label != "neutral":
        print(f"{label}: {count} time(s)")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
report_filename = f"reports/feedback_report_{timestamp}.txt"

with open(report_filename, "w") as f:
    f.write("🎾 MatchMate Feedback Report 🎾\n\n")
    f.write("🔹 Shot Counts:\n")
    f.write(f"Forehands: {shot_counter.nb_backhands}\n")
    f.write(f"Backhands: {shot_counter.nb_forehands}\n")
    f.write(f"Serves: {shot_counter.nb_serves}\n\n")
    f.write("🔹 Feedback Summary:\n")
    for label, count in feedback_stats.items():
        if label != "neutral":
            f.write(f"{label}: {count} time(s)\n")

print(f"\nFeedback report saved to {report_filename} ✅")
