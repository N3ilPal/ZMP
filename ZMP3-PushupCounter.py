import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# For drawing landmarks on the image.
mp_drawing = mp.solutions.drawing_utils

# Define a function to calculate the vertical distance between two points.
def calculate_vertical_distance(point1, point2):
    return abs(point1.y - point2.y)

# Setup video capture.
cap = cv2.VideoCapture(0)

# Variables to keep track of push-up counts.
pushup_counter = 0
last_position = None
pushup_down = False
threshold = 0.05  # Vertical distance change threshold to detect push-up down and up.

print("Please perform push-ups in front of the camera.")

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB, and process it with MediaPipe Pose.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Convert the image back to BGR for OpenCV.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Draw the pose landmarks on the image.
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get landmarks for shoulder and hip.
            shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]

            # Calculate the vertical distance between the shoulder and hip.
            vertical_distance = calculate_vertical_distance(shoulder, hip)

            # Determine if a push-up is being performed.
            if last_position is not None:
                if vertical_distance > last_position + threshold:
                    pushup_down = True
                elif vertical_distance < last_position - threshold and pushup_down:
                    pushup_counter += 1
                    pushup_down = False

            last_position = vertical_distance

        # Display the number of push-ups.
        cv2.putText(image, f'Push-Ups: {pushup_counter}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the image.
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing 'ESC'
            break

finally:
    # Release the video capture object and close all OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()
