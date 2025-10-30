# recognize_cnn.py
import os
import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import urllib.request
import time

# Configuration
DATASET_DIR = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_recognition_model.h5")
ATTENDANCE_LOG = "attendance.csv"

# Download face detection model files if they don't exist
def download_face_detection_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Download deploy.prototxt
    deploy_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    deploy_path = os.path.join(MODEL_DIR, "deploy.prototxt")
    
    if not os.path.exists(deploy_path):
        print("Downloading deploy.prototxt...")
        urllib.request.urlretrieve(deploy_url, deploy_path)
    
    # Download the model weights
    model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    model_path = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
    
    if not os.path.exists(model_path):
        print("Downloading face detection model...")
        urllib.request.urlretrieve(model_url, model_path)
    
    return deploy_path, model_path

# Load face detection model
try:
    deploy_path, model_path = download_face_detection_models()
    FACE_DETECTION_MODEL = cv2.dnn.readNetFromCaffe(deploy_path, model_path)
except Exception as e:
    print(f"Error loading face detection model: {e}")
    print("Please check your internet connection and try again.")
    exit(1)

CONFIDENCE_THRESHOLD = 0.7
RECOGNITION_THRESHOLD = 0.5
RECOGNITION_INTERVAL = 5  # seconds between recognitions

def load_face_embeddings():
    """Load pre-computed face embeddings and labels"""
    embeddings_path = os.path.join(MODEL_DIR, "face_embeddings.npz")
    label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.npz")
    
    if not os.path.exists(embeddings_path) or not os.path.exists(label_encoder_path):
        print("Face embeddings or label encoder not found. Please run train_cnn.py first.")
        return None, None, None
    
    data = np.load(embeddings_path)
    label_data = np.load(label_encoder_path, allow_pickle=True)
    
    return data['embeddings'], data['names'], label_data['classes']

def mark_attendance(name):
    """Mark attendance in the CSV file"""
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

def extract_face_embedding(face_img, model):
    """Extract face embedding using the CNN model"""
    # Preprocess the face image
    face = cv2.resize(face_img, (160, 160))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    
    # Get the embedding from the second-to-last layer
    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-2].output
    )
    embedding = feature_extractor.predict(face, verbose=0)[0]
    return embedding / np.linalg.norm(embedding)  # Normalize

def recognize_faces():
    # Load the face recognition model
    try:
        model = load_model(MODEL_PATH)
        print("Loaded pre-trained face recognition model")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run train_cnn.py first to train the model.")
        return

    # Load known face embeddings
    known_embeddings, known_names, label_classes = load_face_embeddings()
    if known_embeddings is None:
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_recognized = None
    last_recognition_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get frame dimensions
            (h, w) = frame.shape[:2]
            
            # Prepare the frame for face detection
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False
            )
            
            # Detect faces
            FACE_DETECTION_MODEL.setInput(blob)
            detections = FACE_DETECTION_MODEL.forward()

            current_time = time.time()
            
            # Process each detection
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Filter out weak detections
                if confidence > CONFIDENCE_THRESHOLD:
                    # Compute face bounding box
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Ensure the bounding boxes fall within the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                    
                    # Extract the face ROI
                    face = frame[startY:endY, startX:endX]
                    
                    # Only process if the face is large enough
                    if face.shape[0] < 20 or face.shape[1] < 20:
                        continue
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    
                    # Only recognize every few seconds
                    if current_time - last_recognition_time > RECOGNITION_INTERVAL:
                        try:
                            # Get face embedding
                            embedding = extract_face_embedding(face, model)
                            
                            # Compare with known faces
                            similarities = np.array([np.dot(embedding, emb) for emb in known_embeddings])
                            best_match_idx = np.argmax(similarities)
                            best_similarity = similarities[best_match_idx]
                            
                            if best_similarity > RECOGNITION_THRESHOLD:
                                name = known_names[best_match_idx]
                                last_recognized = name
                                last_recognition_time = current_time
                                mark_attendance(name)
                                print(f"Recognized: {name} (Similarity: {best_similarity:.2f})")
                                
                        except Exception as e:
                            print(f"Error during recognition: {e}")
                    
                    # Display the name if recognized
                    if last_recognized and current_time - last_recognition_time < RECOGNITION_INTERVAL:
                        cv2.putText(frame, last_recognized, (startX, startY - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, f"Conf: {best_similarity:.2f}", 
                                   (startX, endY + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
    
    recognize_faces()