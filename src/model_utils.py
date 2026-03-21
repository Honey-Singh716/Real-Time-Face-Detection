import cv2
import numpy as np
import tensorflow as tf
from .config import IMAGE_SIZE, MODEL_PATH
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def load_trained_model():
    """Load the trained mask detection model."""
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_face(face_img):
    face_resized = cv2.resize(face_img, (IMAGE_SIZE, IMAGE_SIZE))
    face_preprocessed = preprocess_input(face_resized.astype(np.float32))
    face_expanded = np.expand_dims(face_preprocessed, axis=0)
    return face_expanded

def predict_mask(model, preprocessed_face):
    """
    Predict mask or no mask.
    Args:
        model: Trained Keras model.
        preprocessed_face: Preprocessed face image.
    Returns:
        Tuple of (label, confidence)
    """
    prediction = model.predict(preprocessed_face, verbose=0)
    prob = prediction[0][0]
    label = "No Mask" if prob > 0.5 else "Mask"
    confidence = prob if label == "Mask" else 1 - prob
    return label, confidence
