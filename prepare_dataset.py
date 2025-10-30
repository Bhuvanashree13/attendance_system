# prepare_dataset.py
import cv2
import os

def capture_images(name, count=30, out_dir='dataset'):
    # Create output directory if it doesn't exist
    person_dir = os.path.join(out_dir, name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Initialize the webcam with lower resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("Press SPACE to capture, ESC to quit.")
    collected = 0
    frame_count = 0
    
    try:
        while collected < count:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only process every other frame to improve performance
            frame_count += 1
            if frame_count % 2 != 0:
                continue
                
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with optimized parameters
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Draw rectangle around the faces on the original frame
            for (x, y, w, h) in faces:
                # Scale back up face locations since the frame was resized
                x, y, w, h = x*2, y*2, w*2, h*2
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow("Enroll: " + name, frame)
            
            # Check for key presses
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC key to quit
                break
            if k == 32:  # SPACE key to capture
                if len(faces) > 0:
                    # Save the face region from the original frame
                    x, y, w, h = [v * 2 for v in faces[0]]  # Scale back up
                    face = frame[y:y+h, x:x+w]
                    if face.size > 0:
                        fname = os.path.join(person_dir, f"{collected+1}.jpg")
                        cv2.imwrite(fname, face)
                        collected += 1
                        print(f"Saved {fname}")
                else:
                    print("No face detected, try again.")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Capture face images for training')
    parser.add_argument("name", help="Name of the person")
    parser.add_argument("--count", type=int, default=30, help="Number of images to capture")
    args = parser.parse_args()
    
    print(f"Starting to capture {args.count} images for {args.name}")
    capture_images(args.name, args.count)