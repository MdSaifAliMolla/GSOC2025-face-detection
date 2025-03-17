import cv2
import numpy as np
import time
import argparse
from deepface import DeepFace
import mediapipe as mp
import threading
import queue

parser = argparse.ArgumentParser()
parser.add_argument("--display_video", type=bool, default=True, help="Whether to display video output")
parser.add_argument("--use_webcam", type=bool, default=True, help="Whether to use webcam instead of video file")
parser.add_argument("--video_path", type=str, default="", help="Path to video file if not using webcam")
parser.add_argument("--detection_interval", type=float, default=0.5, help="How often to run emotion detection (in seconds)")
args = parser.parse_args()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Initialize variables
frame_interval = args.detection_interval 
last_detection_time = 0
result_queue = queue.Queue()
processing_frame = False
current_emotions = []

def analyze_emotions(frame, face_locations):
    """Analyze emotions in separate thread to avoid blocking the main loop"""
    global current_emotions
    
    results = []
    
    for i, face_loc in enumerate(face_locations):
        # Extract face ROI
        x, y, w, h = face_loc
        
        # Ensure coordinates are within frame boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            continue
            
        face_roi = frame[y:y+h, x:x+w]
        
        try:
            # Use DeepFace for emotion analysis
            analysis = DeepFace.analyze(
                img_path=face_roi,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            if isinstance(analysis, list):
                analysis = analysis[0]
                
            emotion = analysis['dominant_emotion']
            emotion_scores = analysis['emotion']
            
            results.append((emotion, emotion_scores))
        except Exception as e:
            print(f"[WARNING] Error analyzing emotions: {e}")
            results.append((None, None))
    
    current_emotions = results

def detect_faces(frame):
    """Detect faces using MediaPipe"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_detection.process(rgb_frame)
    
    face_locations = []
    
    if results.detections:
        frame_height, frame_width = frame.shape[:2]
        
        for detection in results.detections:
            bounding_box = detection.location_data.relative_bounding_box
            
            # Convert relative coordinates to absolute
            x = int(bounding_box.xmin * frame_width)
            y = int(bounding_box.ymin * frame_height)
            w = int(bounding_box.width * frame_width)
            h = int(bounding_box.height * frame_height)
            
            face_locations.append((x, y, w, h))
    
    return face_locations

def main():
    global last_detection_time, processing_frame, current_emotions
    
    if args.use_webcam:
        print("[INFO] Starting video stream from webcam...")
        video_capture = cv2.VideoCapture(0)
    else:
        print(f"[INFO] Starting video from file: {args.video_path}")
        video_capture = cv2.VideoCapture(args.video_path)
    
    time.sleep(1.0)
    
    print("[INFO] Press 'q' to quit")

    print("[INFO] Loading pre-trained models from DeepFace...")
    
    # Pre-warm the model to avoid first-run delay
    try:
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _ = DeepFace.analyze(img_path=dummy_img, actions=['emotion'], enforce_detection=False)
        print("[INFO]========== DeepFace models loaded successfull====================")
    except Exception as e:
        print(f"[WARNING]============Error loading DeepFace models: {e}========================")
    
    emotion_thread = None
    
    while True:
        # Read frame
        ret, frame = video_capture.read()
        
        if not ret:
            print("[INFO]============ End of video stream================")
            break
        
        
        face_locations = detect_faces(frame)
        
        # Run emotion detection at specified intervals
        current_time = time.time()
        if (current_time - last_detection_time) >= frame_interval and not processing_frame and face_locations:
            last_detection_time = current_time
            processing_frame = True
            
            # Start emotion analysis in a separate thread
            if emotion_thread is not None and emotion_thread.is_alive():
                emotion_thread.join(timeout=0.1)
                
            emotion_thread = threading.Thread(
                target=analyze_emotions,
                args=(frame.copy(), face_locations),
                daemon=True
            )
            emotion_thread.start()
            
            # Set a timer to reset processing flag if thread takes too long
            def reset_processing_flag():
                global processing_frame
                processing_frame = False
            
            threading.Timer(2.0, reset_processing_flag).start()
        
        # Draw face bounding boxes and emotion results
        for i, (x, y, w, h) in enumerate(face_locations):
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # If we have emotion results for this face
            if i < len(current_emotions) and current_emotions[i][0] is not None:
                emotion, emotion_scores = current_emotions[i]
                
                # Display dominant emotion
                text = f"{emotion}"
                y_offset = y - 10 if y - 10 > 10 else y + h + 10
                cv2.putText(frame, text, (x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display emotion bars
                if emotion_scores:
                    bar_x = x + w + 10
                    bar_width = 100
                    
                    # Sort emotions by score for better visualization
                    sorted_emotions = sorted(
                        emotion_scores.items(), 
                        key=lambda item: item[1], 
                        reverse=True
                    )
                    
                    # Show top 4 emotions
                    for j, (emo, score) in enumerate(sorted_emotions[:4]):
                        bar_y = y + j * 20
                        # Draw emotion label
                        cv2.putText(frame, f"{emo}: {score:.1f}%", (bar_x, bar_y + 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        # Draw probability bar
                        cv2.rectangle(frame, (bar_x + 120, bar_y + 5), 
                                     (bar_x + 120 + int(score), bar_y + 15),
                                     (255, 0, 0), -1)
        
        # Display frame
        if args.display_video:
            # Display status
            emotion_status = "Processing" if processing_frame else "Ready"
            status_text = f"Face Detection: ON | Emotion Analysis: {emotion_status}"
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            fps_text = f"Detection interval: {frame_interval}s | Faces: {len(face_locations)}"
            cv2.putText(frame, fps_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Emotion Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()
    print("[INFO] Application terminated")

if __name__ == "__main__":
    main()
