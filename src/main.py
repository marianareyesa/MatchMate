import cv2
import mediapipe as mp
import time
from utils.forehand_utils import (
    check_starting_position,
    check_opening_position,
    check_hitting,
    follow_through,
    normalize_landmarks,
)
from utils.speech_utils import give_feedback, is_speaking
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
min_detection_confidence = float(os.getenv("MIN_DETECTION_CONFIDENCE", 0.5))
min_tracking_confidence = float(os.getenv("MIN_TRACKING_CONFIDENCE", 0.5))

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=min_detection_confidence, 
                    min_tracking_confidence=min_tracking_confidence)

# OpenCV video capture
cap = cv2.VideoCapture(0)

# State management
stage = "starting_position"  # Initial stage
last_feedback_time = time.time()  # Track feedback timing
feedback_cooldown = 5  # Wait before giving new feedback
transition_delay = 2  # Hold perfect form for 5 seconds before transition

# Visual confirmation variables
perfect_position_start = None  # Time when player first reaches perfect position
perfect_feedback_given = False  # Ensure feedback is played once
show_green_overlay = False  # Controls the green overlay effect

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Draw pose landmarks
        if show_green_overlay:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=5),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=5))
        else:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Normalize landmarks
        normalized_landmarks = normalize_landmarks(results.pose_landmarks.landmark, frame_width, frame_height)

        current_time = time.time()

        def process_stage(stage_name, check_function, next_stage, success_message):
            """General function to process each stage of the forehand."""
            global perfect_position_start, perfect_feedback_given, show_green_overlay, stage, last_feedback_time

            feedback = check_function(normalized_landmarks)

            if feedback:
                if current_time - last_feedback_time >= feedback_cooldown:
                    for fb in feedback:
                        give_feedback(fb)  # Runs in parallel
                    last_feedback_time = current_time
                perfect_position_start = None  # Reset timer if incorrect
                perfect_feedback_given = False  # Reset flag
                show_green_overlay = False  # Remove green overlay
            else:
                if perfect_position_start is None:  # Start the timer only once
                    perfect_position_start = time.time()
                    perfect_feedback_given = False  # Reset flag

                # Ensure "Perfect" feedback is only spoken once
                if not perfect_feedback_given and not is_speaking:
                    give_feedback(success_message)
                    perfect_feedback_given = True  # Mark as given

                # Keep green overlay active
                show_green_overlay = True

                # If 5 seconds have passed in perfect form AND speech has finished, move to next stage
                if time.time() - perfect_position_start >= transition_delay and not is_speaking:
                    show_green_overlay = False
                    perfect_feedback_given = False  # Reset for next stage
                    stage = next_stage

        # --- Step 1: Check Starting Position ---
        if stage == "starting_position":
            process_stage("starting_position", check_starting_position, "opening_position",
                          "Perfect! You are in the correct starting position. Let's move to the next step, the opening position.")

        # --- Step 2: Check Opening Position ---
        elif stage == "opening_position":
            process_stage("opening_position", check_opening_position, "hitting_position",
                          "Great! You are in the correct opening position. Now prepare for the hit.")

        # --- Step 3: Check Hitting Position ---
        elif stage == "hitting_position":
            process_stage("hitting_position", check_hitting, "follow_through",
                          "Nice shot! Now follow through properly.")

        # --- Step 4: Check Follow-Through Position ---
        elif stage == "follow_through":
            process_stage("follow_through", follow_through, "completed",
                          "Excellent follow-through! Forehand completed successfully.")

    # Display current stage
    cv2.putText(frame, f"Stage: {stage.replace('_', ' ').title()}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if show_green_overlay:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (0, 255, 0), -1)
        alpha = 0.3  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.imshow("Forehand Position Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()