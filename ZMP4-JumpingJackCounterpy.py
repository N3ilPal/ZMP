import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# For drawing landmarks on the image.
mp_drawing = mp.solutions.drawing_utils

# Define a function to calculate the horizontal distance between two points.
def calculate_horizontal_distance(point1, point2):
    return abs(point1.x - point2.x)

# Define a function to calculate the vertical distance between two points.
def calculate_vertical_distance(point1, point2):
    return abs(point1.y - point2.y)

# Setup video capture.
cap = cv2.VideoCapture(0)

# Variables to keep track of jumping jacks.
jumping_jack_counter = 0
legs_apart = False
arms_up = False
threshold = 0.1  # Distance change threshold to detect movement.

print("Please perform jumping jacks in front of the camera.")

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

            # Get landmarks for legs and arms.
            left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
            left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Calculate horizontal distance between the ankles.
            leg_distance = calculate_horizontal_distance(left_ankle, right_ankle)
            # Calculate vertical distance between the wrists and shoulders.
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            arm_distance = (calculate_vertical_distance(left_wrist, left_shoulder) + calculate_vertical_distance(right_wrist, right_shoulder)) / 2

            # Determine if a jumping jack is being performed.
            if leg_distance > threshold and arm_distance > threshold:
                if not legs_apart:
                    legs_apart = True
                    arms_up = True
            elif leg_distance < threshold and arm_distance < threshold:
                if legs_apart and arms_up:
                    jumping_jack_counter += 1
                    legs_apart = False
                    arms_up = False

        # Display the number of jumping jacks.
        cv2.putText(image, f'Jumping Jacks: {jumping_jack_counter}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the image.
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing 'ESC'
            break

finally:
    # Release the video capture object and close all OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()
