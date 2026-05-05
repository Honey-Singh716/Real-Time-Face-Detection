import cv2
import json
import time
import os
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from .model_utils import load_trained_model


def main():
     
    # Load Mask Classification Model
     
    try:
        model = load_trained_model()
        print("Mask model loaded successfully.")
    except Exception as e:
        print(f"Error loading mask model: {e}")
        return

    # Confirm model output logic
    print("Reminder: Sigmoid output = probability of class index 1")
     
    # Load DNN Face Detector (ABSOLUTE PATH FIX)
     
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prototxt_path = os.path.join(BASE_DIR, "models", "deploy.prototxt")
        model_path = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")

        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        print("DNN face detector loaded successfully.")
        detector_type = 'dnn'
    except Exception as e:
        print(f"DNN face detector failed: {e}. Falling back to Haar Cascade.")
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if face_cascade.empty():
            print("Error loading Haar cascade.")
            return
        print("Haar Cascade loaded successfully.")
        detector_type = 'haar'

     
    # Start Webcam
     
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    # Load optimal threshold
    try:
        with open('models/optimal_threshold.txt', 'r') as f:
            optimal_threshold = float(f.read().strip())
    except:
        optimal_threshold = 0.5
    print(f"Using threshold: {optimal_threshold}")

    # Load class indices
    try:
        with open('models/class_indices.json', 'r') as f:
            class_indices = json.load(f)
        positive_class = max(class_indices, key=class_indices.get)  # Class with index 1
        print(f"Positive class (index 1): {positive_class}")
    except:
        positive_class = 'mask'  # Default
        print("Could not load class_indices, using default positive class: mask")

    # Parameters

    confidence_threshold = 0.6
    frame_skip = 1
    frame_count = 0
    alpha = 0.5
    history_dict = {}  # Dict of deques per face ID (centroid)

    fps = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for faster processing
        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]
        frame_count += 1

        # Apply CLAHE for low light
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

         
        # Face Detection (Every N Frames)
         
        if detector_type == 'dnn':
            if frame_count % frame_skip == 0:

                blob = cv2.dnn.blobFromImage(
                    frame,
                    1.0,
                    (300, 300),
                    (104.0, 177.0, 123.0)
                )

                net.setInput(blob)
                detections = net.forward()

                faces = []

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    if confidence > confidence_threshold:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        startX, startY, endX, endY = box.astype("int")

                        # Clamp bbox
                        startX = max(0, startX)
                        startY = max(0, startY)
                        endX = min(w, endX)
                        endY = min(h, endY)

                        if (endX - startX) > 80 and (endY - startY) > 80:
                            faces.append((startX, startY, endX, endY))

                if faces:
                    # Process each face
                    for startX, startY, endX, endY in faces:
                        # Clamp
                        startX = max(0, startX)
                        startY = max(0, startY)
                        endX = min(w, endX)
                        endY = min(h, endY)

                        # Compute centroid for ID
                        center_x = ((startX + endX) // 2) // 10 * 10
                        center_y = ((startY + endY) // 2) // 10 * 10
                        face_id = (center_x, center_y)

                        face_roi = frame[startY:endY, startX:endX]

                        if face_roi.size > 0 and (endX - startX) > 50 and (endY - startY) > 50:
                            # Crop lower half
                            h_roi = face_roi.shape[0]
                            face_roi = face_roi[h_roi//2:, :]
                            face_resized = cv2.resize(face_roi, (224, 224))
                            face_preprocessed = preprocess_input(face_resized.astype(np.float32))
                            face_expanded = np.expand_dims(face_preprocessed, axis=0)

                            prediction = model.predict(face_expanded, verbose=0)[0][0]

                            # Dynamic label interpretation based on positive class
                            if positive_class == 'mask':
                                if prediction > optimal_threshold:
                                    label = "Mask"
                                    confidence = float(prediction)
                                else:
                                    label = "No Mask"
                                    confidence = float(1 - prediction)
                            else:
                                if prediction > optimal_threshold:
                                    label = "No Mask"
                                    confidence = float(prediction)
                                else:
                                    label = "Mask"
                                    confidence = float(1 - prediction)

                            # Get or create history for this face
                            if face_id not in history_dict:
                                history_dict[face_id] = deque(maxlen=3)
                            history_dict[face_id].append(label)

                            # Majority vote for stability
                            hist = history_dict[face_id]
                            if len(hist) >= 3:
                                mask_count = hist.count("Mask")
                                stable_label = "Mask" if mask_count >= 2 else "No Mask"
                            else:
                                stable_label = label

                            color = (0, 255, 0) if stable_label == "Mask" else (0, 0, 255)

                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                            cv2.putText(
                                frame,
                                f"{stable_label}: {confidence:.2f}",
                                (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                color,
                                2
                            )
        else:
            # Haar Cascade fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            if len(faces) > 0:
                # Process each face
                for (x, y, w_h, h_h) in faces:
                    startX, startY, endX, endY = x, y, x + w_h, y + h_h

                    # Clamp
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    # Compute centroid for ID
                    center_x = ((startX + endX) // 2) // 10 * 10
                    center_y = ((startY + endY) // 2) // 10 * 10
                    face_id = (center_x, center_y)

                    face_roi = frame[startY:endY, startX:endX]

                    if face_roi.size > 0 and (endX - startX) > 50 and (endY - startY) > 50:
                        # Crop lower half
                        h_roi = face_roi.shape[0]
                        face_roi = face_roi[h_roi//2:, :]
                        face_resized = cv2.resize(face_roi, (224, 224))
                        face_preprocessed = preprocess_input(face_resized.astype(np.float32))
                        face_expanded = np.expand_dims(face_preprocessed, axis=0)

                        prediction = model.predict(face_expanded, verbose=0)[0][0]

                        # Dynamic label interpretation based on positive class
                        if positive_class == 'with_mask':
                            if prediction > optimal_threshold:
                                label = "Mask"
                                confidence = float(prediction)
                            else:
                                label = "No Mask"
                                confidence = float(1 - prediction)
                        else:
                            if prediction > optimal_threshold:
                                label = "No Mask"
                                confidence = float(prediction)
                            else:
                                label = "Mask"
                                confidence = float(1 - prediction)

                        # Get or create history for this face
                        if face_id not in history_dict:
                            history_dict[face_id] = deque(maxlen=3)
                        history_dict[face_id].append(label)

                        # Majority vote for stability
                        hist = history_dict[face_id]
                        if len(hist) >= 3:
                            mask_count = hist.count("Mask")
                            stable_label = "Mask" if mask_count >= 2 else "No Mask"
                        else:
                            stable_label = label

                        color = (0, 255, 0) if stable_label == "Mask" else (0, 0, 255)

                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                        cv2.putText(
                            frame,
                            f"{stable_label}: {confidence:.2f}",
                            (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            color,
                            2
                        )

         
        # FPS (Smoothed)
         
        curr_time = time.time()
        instant_fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        fps = 0.9 * fps + 0.1 * instant_fps if fps != 0 else instant_fps

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.imshow("Face Mask Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()