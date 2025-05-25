# face_analysis.py

import cv2
import dlib
import numpy as np
from utils import euclidean_distance

# Initialize dlib's face detector and shape predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(predictor_path)
except RuntimeError:
    raise RuntimeError(f"Required shape predictor not found. Please download 'shape_predictor_68_face_landmarks.dat' and place it in the working directory.")

def detect_landmarks(image):
    """
    Detect facial landmarks for the first detected face in the image.
    Returns a NumPy array of (x, y) coordinates for 68 landmarks.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        raise Exception("No face detected in the image.")
    face = faces[0]  # consider the first detected face
    shape = predictor(gray, face)
    landmarks = np.array([(pt.x, pt.y) for pt in shape.parts()])
    return landmarks

def classify_face_shape(landmarks):
    """
    Classify face shape based on geometric measurements of facial landmarks.
    Returns one of: round, oval, square, heart, diamond, triangle, oblong.
    """
    # Key landmark indices:
    # Chin (8), jaw corners (4, 12), cheeks (3, 13), eyebrows (17, 26)
    chin_y = landmarks[8][1]
    # Forehead approximation: use the top of the eyebrows (min Y of indices 17-26)
    brow_y = min(landmarks[i][1] for i in range(17, 27))
    face_length = chin_y - brow_y

    # Horizontal distances
    left_jaw = landmarks[0]
    right_jaw = landmarks[16]
    jaw_width = euclidean_distance(landmarks[4], landmarks[12])
    cheek_width = euclidean_distance(landmarks[3], landmarks[13])
    forehead_width = euclidean_distance(landmarks[17], landmarks[26])
    face_width = euclidean_distance(left_jaw, right_jaw)

    # Avoid division by zero
    if face_width == 0:
        return "undefined"

    ratio = face_length / face_width

    # Classify based on relative proportions
    if ratio > 1.5:
        shape = "oblong"
    elif ratio > 1.15:
        shape = "oval"
    else:
        # Widths comparison for other shapes
        f = forehead_width
        c = cheek_width
        j = jaw_width
        if f > c and f > j and j < c:
            shape = "heart"
        elif j > f and j > c and f < c:
            shape = "triangle"
        elif c > f and c > j:
            shape = "diamond"
        else:
            # Square vs Round
            if abs(j - c) < 0.1 * face_width:
                shape = "square"
            else:
                shape = "round"
    return shape
