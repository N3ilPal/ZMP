import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize the Video Capture
cap = cv2.VideoCapture(0)

# Variables to keep track of faces and counter
face_in_frame = False
counter = 0

with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform face detection
        results = face_detection.process(rgb_frame)
        
        # Check if any faces are detected
        if results.detections:
            if not face_in_frame:
                counter += 1
                face_in_frame = True
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
        else:
            face_in_frame = False
        
        # Display the frame
        cv2.putText(frame, f'Face Entries: {counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
