from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import time

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def detect_emotion_from_video(capture_duration=5):
    # Start video streaming
    cv2.namedWindow('Emotion Detection')
    camera = cv2.VideoCapture(0)  # Start webcam
    start_time = time.time()

    # Dictionary to store cumulative probabilities for each emotion
    emotion_probs = {emotion: 0 for emotion in EMOTIONS}

    while int(time.time() - start_time) < capture_duration:
        ret, frame = camera.read()  # Capture frame-by-frame
        if not ret:
            break  # If frame not captured properly, break

        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        frameClone = frame.copy()

        if len(faces) > 0:
            # Sort faces by size, focusing on the largest one
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict the emotion
            preds = emotion_classifier.predict(roi)[0]

            # Add probabilities to cumulative emotion probabilities
            for i, emotion in enumerate(EMOTIONS):
                emotion_probs[emotion] += preds[i]

            # Display the label on the frame
            label = EMOTIONS[preds.argmax()]
            cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        # Show the frame with the detected face and label in real-time
        cv2.imshow('Emotion Detection', frameClone)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

    # Find the emotion with the highest cumulative probability
    dominant_emotion = max(emotion_probs, key=emotion_probs.get)
    return dominant_emotion  # Return the dominant emotion
"""
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import time

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def detect_emotion_from_video(capture_duration=10):
    # Start video streaming
    cv2.namedWindow('lets Detect your emotion')
    camera = cv2.VideoCapture(0)
    start_time = time.time()

    # Dictionary to store cumulative probabilities for each emotion
    emotion_probs = {emotion: 0 for emotion in EMOTIONS}

    while int(time.time() - start_time) < capture_duration:
        frame = camera.read()[1]
        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        frameClone = frame.copy()
        
        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            preds = emotion_classifier.predict(roi)[0]
            
            # Add probabilities to cumulative emotion probabilities
            for i, emotion in enumerate(EMOTIONS):
                emotion_probs[emotion] += preds[i]

    # Release the camera
        cv2.imshow('Lets detect your emotion',frameClone)
    camera.release()
    cv2.destroyAllWindows()

    # Find the emotion with the highest cumulative probability
    dominant_emotion = max(emotion_probs, key=emotion_probs.get)
    return dominant_emotion  # Return the dominant emotion

"""