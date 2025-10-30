# recognize_and_log.py
import os
import cv2
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder

# Configuration
DATASET_DIR = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_recognition_model.h5")
ATTENDANCE_LOG = "attendance.csv"
IMG_SIZE = (160, 160)
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to consider a match

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained model and label encoder
def load_trained_model():
    # Load the model
    if not os.path.exists(MODEL_PATH):
        print("Trained model not found. Please train the model first.")
        return None, None, None
    
    model = load_model(MODEL_PATH)
    
    # Load label encoder classes
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(os.path.join(MODEL_DIR, 'label_encoder.npy'), allow_pickle=True)
    
    # Create feature extractor (model without the last layer)
    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-2].output
    )
    
    return model, feature_extractor, label_encoder

def preprocess_face(face_img):
    # Resize and preprocess the face image for the model
    face_img = cv2.resize(face_img, IMG_SIZE)
    face_img = img_to_array(face_img)
    face_img = preprocess_input(face_img)
    return np.expand_dims(face_img, axis=0)

def mark_attendance(name):
    with open(ATTENDANCE_LOG, 'a+') as f:
        f.seek(0)
        existing = f.read()
        now = datetime.now()
        date_string = now.strftime('%Y-%m-%d')
        time_string = now.strftime('%H:%M:%S')
        
        entry = f"{name},{date_string},{time_string}\n"
        
        if name not in existing:
            f.write(entry)
            print(f"Attendance marked for {name} at {time_string}")

def recognize_faces():
    # Load the trained model and label encoder
    model, feature_extractor, label_encoder = load_trained_model()
    if model is None:
        return
    
    # Get class names from label encoder
    class_names = label_encoder.classes_
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Dictionary to store recognition info for each face
    face_records = {}
    recognition_interval = 2  # seconds between recognitions

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            current_time = time.time()
            
            # Process each face
            for i, (x, y, w, h) in enumerate(faces):
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Initialize or get face record
                if i not in face_records:
                    face_records[i] = {
                        'last_recognition_time': 0,
                        'name': 'Unknown',
                        'best_score': 0
                    }
                
                face_record = face_records[i]
                
                # Only recognize every few seconds for each face
                if current_time - face_record['last_recognition_time'] > recognition_interval:
                    # Extract and preprocess face
                    face_img = frame[y:y+h, x:x+w]
                    if face_img.size > 0:
                        # Preprocess the face for the model
                        processed_face = preprocess_face(face_img)
                        
                        # Get prediction
                        predictions = model.predict(processed_face, verbose=0)
                        confidence = np.max(predictions)
                        predicted_class = np.argmax(predictions, axis=1)
                        
                        # Only accept predictions above confidence threshold
                        if confidence > CONFIDENCE_THRESHOLD:
                            name = label_encoder.inverse_transform(predicted_class)[0]
                            best_score = float(confidence)
                        else:
                            name = "Unknown"
                            best_score = 0.0
                        
                        # Update face record
                        face_record['name'] = name
                        face_record['best_score'] = best_score
                        face_record['last_recognition_time'] = current_time
                        
                        # Only mark attendance for known faces with high confidence
                        if name != "Unknown":
                            mark_attendance(name)
                            print(f"Recognized: {name} (Confidence: {best_score:.2f})")
                
                # Display the name with different colors for known/unknown
                name = face_record['name']
                color = (36, 255, 12) if name != "Unknown" else (0, 0, 255)  # Green for known, Red for unknown
                cv2.putText(frame, name, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Clean up old face records (to prevent memory leaks)
            face_records = {k: v for k, v in face_records.items() 
                          if current_time - v['last_recognition_time'] < recognition_interval * 2}

            # Display the frame
            cv2.imshow('Face Recognition', frame)
            
            # Break the loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create attendance log file if it doesn't exist
    if not os.path.exists(ATTENDANCE_LOG):
        with open(ATTENDANCE_LOG, 'w') as f:
            f.write("Name,Date,Time\n")
    
    import time
    recognize_faces()