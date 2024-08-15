import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose and Hands.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# For drawing landmarks on the image.
mp_drawing = mp.solutions.drawing_utils

# Define a function to calculate the vertical distance between two points.
def calculate_vertical_distance(point1, point2):
    return abs(point1.y - point2.y)

# Setup video capture.
cap = cv2.VideoCapture(0)

# Variables to keep track of dumbbell lifts.
left_arm_counter = 0
right_arm_counter = 0
total_counter = 0

left_arm_up = False
right_arm_up = False

threshold = 0.1  # Vertical distance change threshold to detect lift.

print("Please lift dumbbells to your chest one by one in front of the camera.")

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Invert the image (flip horizontally).
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB, and process it with MediaPipe Pose and Hands.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(image_rgb)
        hands_results = hands.process(image_rgb)

        # Convert the image back to BGR for OpenCV.
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks on the image.
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get landmarks for wrists and shoulders.
            left_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # Calculate vertical distances from wrists to shoulders.
            left_arm_distance = calculate_vertical_distance(left_wrist, left_shoulder)
            right_arm_distance = calculate_vertical_distance(right_wrist, right_shoulder)

            # Determine if a lift is being performed.
            if left_arm_distance < threshold:
                if not left_arm_up:
                    left_arm_counter += 1
                    total_counter += 1
                    left_arm_up = True
            else:
                left_arm_up = False

            if right_arm_distance < threshold:
                if not right_arm_up:
                    right_arm_counter += 1
                    total_counter += 1
                    right_arm_up = True
            else:
                right_arm_up = False

        # Draw hand landmarks on the image.
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the number of lifts.
        cv2.putText(image, f'Left Arm Lifts: {left_arm_counter}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Right Arm Lifts: {right_arm_counter}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Total Lifts: {total_counter}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the image with the skeleton overlay.
        cv2.imshow('MediaPipe Pose and Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing 'ESC'
            break

finally:
    # Release the video capture object and close all OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()
