import cv2
import mediapipe as mp

# Initialize mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(r'C:\Users\karti\OneDrive\Desktop\My Work\Projects\Qatar project\1723015384578.mp4')

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detection.process(img_rgb)

        # Draw detections on the frame
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        # Display the frame
        cv2.imshow('Face Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()