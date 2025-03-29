import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def normalize_landmarks(landmarks, frame_width, frame_height):
    """
    Normalize landmarks to pixel coordinates and scale based on the frame dimensions.
    Args:
        landmarks: List of MediaPipe landmarks.
        frame_width: Width of the video frame.
        frame_height: Height of the video frame.
    Returns:
        dict: Normalized landmarks with pixel coordinates.
    """
    normalized_landmarks = {
        i: (landmark.x * frame_width, landmark.y * frame_height)
        for i, landmark in enumerate(landmarks)
    }
    return normalized_landmarks

def intro_forehand():
    pass

def check_starting_position(landmarks):
    """Check if the player is in the correct starting position."""
    feedback = []

    # Calculate knee angle (left leg)
    left_knee_angle = calculate_angle(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
    )
    if not (120 <= left_knee_angle <= 165):
        feedback.append(
            f"Your left knee is bent at {left_knee_angle:.1f} degrees. "
            + ("Bend your left knee more." if left_knee_angle > 165 else "Straighten your left knee slightly.")
        )

    # Calculate knee angle (right leg)
    right_knee_angle = calculate_angle(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
    )
    if not (120 <= right_knee_angle <= 165):
        feedback.append(
            f"Your right knee is bent at {right_knee_angle:.1f} degrees. "
            + ("Bend your right knee more." if right_knee_angle > 165 else "Straighten your right knee slightly.")
        )

    # Check stance width (distance between feet)
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    stance_width = abs(left_ankle[0] - right_ankle[0])
    hip_width = abs(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0] - landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][0]
    )
    if stance_width < 1.2 * hip_width:
        feedback.append(f"Your stance width is {stance_width:.2f} units, which is too narrow. Widen your stance slightly.")

    # Check arm position (racket in front & at belly button level)
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Get midpoints
    shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
    shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_mid_x = (left_hip[0] + right_hip[0]) / 2
    hip_mid_y = (left_hip[1] + right_hip[1]) / 2  # Belly button level is around hip height

    # Racket (wrists midpoint)
    racket_mid_x = (left_wrist[0] + right_wrist[0]) / 2
    racket_mid_y = (left_wrist[1] + right_wrist[1]) / 2

    # Provide explicit racket positioning instructions
    if racket_mid_y < hip_mid_y - 20:
        feedback.append("Lower your racket")
    elif racket_mid_y > hip_mid_y + 20:
        feedback.append("Lift your racket higher")

    # ✅ Improved: Check torso posture (upright position)
    torso_angle = calculate_angle(left_hip, left_shoulder, right_shoulder)
    if not (80 <= torso_angle <= 100):
        feedback.append(
            f"Your torso angle is {torso_angle:.1f} degrees. Keep your torso upright."
        )

    return feedback

def check_opening_position(landmarks):
    """Check if the player is in the correct opening position."""
    feedback = []

    # Check left arm alignment (should be straight and aligned with the shoulder)
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    if not (160 <= left_arm_angle <= 180):
        feedback.append(f"Your left arm is at {left_arm_angle:.1f} degrees. Keep it straighter and aligned with your shoulder.")

    # Check right arm alignment (should be straight)
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    if not (160 <= right_arm_angle <= 180):
        feedback.append(f"Your right arm is at {right_arm_angle:.1f} degrees. Keep it straighter.")

    # Check if both arms are aligned
    shoulder_alignment_angle = calculate_angle(left_shoulder, right_shoulder, right_wrist)
    if not (160 <= shoulder_alignment_angle <= 180):
        feedback.append(f"Your arms are not properly aligned. Adjust them to be in line with each other.")

    # Check racket position (should be upright)
    racket_tip = (right_wrist[0], right_wrist[1] - 50)  # Simulated racket tip
    racket_angle = calculate_angle(right_elbow, right_wrist, racket_tip)

    if not (85 <= racket_angle <= 100):  # Racket should be vertical
        feedback.append("Keep your racket upright. Don't drop it.")

    # Check knee position (both should be bent)
    left_knee_angle = calculate_angle(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
    )
    if not (120 <= left_knee_angle <= 165):
        feedback.append(f"Your left knee is at {left_knee_angle:.1f} degrees. Bend it more for better balance.")

    right_knee_angle = calculate_angle(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
    )
    if not (120 <= right_knee_angle <= 165):
        feedback.append(f"Your right knee is at {right_knee_angle:.1f} degrees. Bend it more.")

    # ✅ 6. Check stance width (should be wide enough)
    left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
    right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

    stance_width = abs(left_foot[0] - right_foot[0])
    hip_width = abs(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0] - landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][0])

    if stance_width < 1.2 * hip_width:
        feedback.append("Widen your stance slightly for better balance.")

    return feedback

def check_hitting(landmarks):
    """Check if the player is in the correct hitting position."""
    feedback = []

    # Check right arm alignment (should be extended but slightly bent)
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    if not (150 <= right_arm_angle <= 170):  # Should be extended but slightly bent
        feedback.append(
            f"Your right arm is at {right_arm_angle:.1f} degrees. Keep it extended but slightly bent."
        )

    # Check hip rotation (should be slightly forward, around 60-100°)
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    hip_rotation_angle = calculate_angle(left_hip, right_hip, right_shoulder)

    if not (60 <= hip_rotation_angle <= 100):  # Should be slightly rotated forward
        feedback.append(
            f"Your hips are at {hip_rotation_angle:.1f} degrees. Rotate your hips slightly forward for better shot power."
        )

    # Check foot positioning
    right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
    left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

    # **Right foot should be behind and pushing off**
    if right_foot[0] > left_foot[0] - 20:  # Allow slight variation
        feedback.append("Keep your right foot behind for better balance and rotation.")

    # **Left foot should be planted forward**
    if left_foot[1] < left_ankle[1] - 10:  # If lifted too much
        feedback.append("Keep your left foot planted for stability.")

    return feedback

def follow_through(landmarks):
    feedback = []

    # Check hip rotation (should be rotated to the left, around 200-250°)
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

    hip_rotation_angle = calculate_angle(right_hip, left_hip, left_shoulder)

    if not (80 <= hip_rotation_angle <= 100):  # Should be rotated left
        feedback.append(
            f"Your hips are at {hip_rotation_angle:.1f} degrees. Rotate your hips more to the left for a full follow-through."
        )

    # Check right wrist positioning (should be around or above shoulder level)
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    if not (right_shoulder[1] - 20 <= right_wrist[1] <= right_shoulder[1] + 20):  # Allow small variation
        if right_wrist[1] > right_shoulder[1]:
            feedback.append("Raise your right wrist slightly higher to complete the follow-through.")
        else:
            feedback.append("Lower your right wrist slightly to be at shoulder level.")

    # Check right elbow positioning (should be around chin level)
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

    if not (right_shoulder[1] - 15 <= right_elbow[1] <= right_shoulder[1] + 20):  # Allow slight variation
        feedback.append("Keep your right elbow close to chin level to maintain proper follow-through form.")

    return feedback
