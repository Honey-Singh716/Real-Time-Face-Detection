import os
import cv2
import json
import numpy as np
import tensorflow as tf
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
from src.model_utils import load_trained_model, preprocess_face
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# SET PAGE CONFIG
st.set_page_config(page_title="Face Mask Detector", page_icon="😷")

# LOAD MODELS
@st.cache_resource
def get_models():
    mask_model = load_trained_model()
    
    # Load DNN Face Detector
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prototxt_path = os.path.join(base_dir, "models", "deploy.prototxt")
    caffemodel_path = os.path.join(base_dir, "models", "res10_300x300_ssd_iter_140000.caffemodel")
    
    if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
        st.error("Face detector models missing in models/ directory!")
        return mask_model, None
        
    face_net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    return mask_model, face_net

# LOAD THRESHOLD & CLASSES
@st.cache_data
def get_config():
    try:
        with open('models/optimal_threshold.txt', 'r') as f:
            threshold = float(f.read().strip())
    except:
        threshold = 0.5
        
    try:
        with open('models/class_indices.json', 'r') as f:
            indices = json.load(f)
        # Assuming index 1 is the positive class
        pos_class = max(indices, key=indices.get) 
    except:
        pos_class = 'mask'
        
    return threshold, pos_class

# VIDEO PROCESSOR
class VideoProcessor(VideoProcessorBase):
    def __init__(self, mask_model, face_net, threshold, pos_class, conf_threshold):
        self.mask_model = mask_model
        self.face_net = face_net
        self.threshold = threshold
        self.pos_class = pos_class
        self.conf_threshold = conf_threshold

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        
        # Face Detection
        # blobFromImage automatically resizes, so we drop the redundant cv2.resize() call
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure box is within frame
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                
                # Classifier ROI
                face_roi = img[startY:endY, startX:endX]
                if face_roi.size > 0:
                    # MobileNetV2 Preprocessing
                    face_resized = cv2.resize(face_roi, (224, 224))
                    face_preprocessed = preprocess_input(face_resized.astype(np.float32))
                    face_expanded = np.expand_dims(face_preprocessed, axis=0)

                    # Prediction
                    # model(..., training=False) is significantly faster for single frames than model.predict()
                    pred = float(self.mask_model(face_expanded, training=False)[0][0])
                    
                    # Logic based on positive class
                    if self.pos_class == 'mask':
                        label = "Mask" if pred > self.threshold else "No Mask"
                        label_conf = pred if label == "Mask" else 1 - pred
                    else:
                        label = "No Mask" if pred > self.threshold else "Mask"
                        label_conf = pred if label == "No Mask" else 1 - pred

                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                    
                    # DRAW LABEL BOX (Solid background)
                    label_text = f"{label}: {label_conf*100:.1f}%"
                    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Ensure label box stays in frame
                    ty = max(startY, th + 10)
                    cv2.rectangle(img, (startX, ty - th - 10), (startX + tw, ty), color, -1)
                    cv2.putText(img, label_text, (startX, ty - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # BORDER
                    cv2.rectangle(img, (startX, startY), (endX, endY), color, 3)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("😷 Real-Time Face Mask Detection")
    st.markdown("""
    This app detects face masks in real-time. Click **START** to begin.
    """)

    mask_model, face_net = get_models()
    opt_threshold, pos_class = get_config()

    if face_net is None:
        st.stop()

    # Default face detection confidence
    conf_threshold = 0.5

    # RTC CONFIG (For STUN servers in deployment)
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="mask-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: VideoProcessor(
            mask_model, face_net, opt_threshold, pos_class, conf_threshold
        ),
        async_processing=True,
    )

    st.markdown("---")

if __name__ == "__main__":
    main()
