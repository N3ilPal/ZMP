import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import threading
import time

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# For drawing landmarks on the image.
mp_drawing = mp.solutions.drawing_utils

# Define a function to calculate the distance between two points.
def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)

# Setup video capture and processing in a separate thread.
def video_capture(frame_placeholder):
    cap = cv2.VideoCapture(0)

    # Variables to keep track of hand opening and closing.
    closing_counter = 0
    last_average_distance = None
    hand_open = False
    hand_close = False
    threshold = 0.1  # Distance change threshold to detect open/close.

    st.write("Please move your hand around to measure hand openings and closings.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            st.write("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB, and process it with MediaPipe Hands.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Convert the image back to BGR for OpenCV.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand landmarks on the image.
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate average distance from wrist to fingertips.
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                fingertips = [hand_landmarks.landmark[i] for i in [mp_hands.HandLandmark.THUMB_TIP, 
                                                                   mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                                                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                                                                   mp_hands.HandLandmark.RING_FINGER_TIP,
                                                                   mp_hands.HandLandmark.PINKY_TIP]]
                distances = [calculate_distance(wrist, fingertip) for fingertip in fingertips]
                average_distance = sum(distances) / len(distances)

                # Determine if the hand is opening or closing.
                if last_average_distance is not None:
                    if average_distance - last_average_distance > threshold:
                        hand_open = True
                        hand_close = False
                    elif last_average_distance - average_distance > threshold:
                        hand_close = True
                        hand_open = False

                    if hand_close and not hand_open:
                        closing_counter += 1
                        hand_close = False  # Reset to avoid multiple counts for the same close action

                last_average_distance = average_distance

        # Display the number of hand closings.
        cv2.putText(image, f'Hand Closings: {closing_counter}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the image.
        frame_placeholder.image(image, channels="BGR")

        if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing 'ESC'
            break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit interface
st.title("Hand Opening/Closing Tracker")

frame_placeholder = st.empty()

# Start video capture in a separate thread
video_thread = threading.Thread(target=video_capture, args=(frame_placeholder,))
video_thread.start()

# Add a stop button
if st.button("Stop"):
    cap.release()
    cv2.destroyAllWindows()
    st.stop()
