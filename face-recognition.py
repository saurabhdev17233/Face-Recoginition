import cv2
import face_recognition
import numpy as np

def face_recognition_system():
    # Load a sample image and learn its features
    known_image = face_recognition.load_image_file("sample.jpg")
    known_encoding = face_recognition.face_encodings(known_image)[0]
    known_faces = [known_encoding]
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces and get encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"
            
            if True in matches:
                name = "Recognized Face"
            
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Face Recognition", frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #saurabhdev17233
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_recognition_system()
