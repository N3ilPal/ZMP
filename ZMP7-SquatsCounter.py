import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define a function to calculate the angle between three points.
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])  # First point
    b = np.array([b.x, b.y])  # Mid point
    c = np.array([c.x, c.y])  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

# Setup video capture.
cap = cv2.VideoCapture(0)

# Variables to keep track of squats.
squat_counter = 0
squat_position = False  # True when the person is in the down position of a squat
hip_knee_threshold = 140  # Angle threshold to detect squat
knee_ankle_threshold = 60  # Angle threshold to detect squat

print("Please perform squats in front of the camera.")

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
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark

            # Get the coordinates.
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Calculate angles.
            left_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_angle = calculate_angle(right_hip, right_knee, right_ankle)
            hip_angle = (calculate_angle(left_knee, left_hip, right_hip) + calculate_angle(right_knee, right_hip, left_hip)) / 2

            # Check if the person is in the squat position.
            if left_angle < hip_knee_threshold and right_angle < hip_knee_threshold and hip_angle > knee_ankle_threshold:
                if not squat_position:
                    squat_position = True
            else:
                if squat_position:
                    squat_counter += 1
                    squat_position = False

        # Display the number of squats.
        cv2.putText(image, f'Squats: {squat_counter}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the image.
        cv2.imshow('Squat Counter', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing 'ESC'
            break

finally:
    # Release the video capture object and close all OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()
