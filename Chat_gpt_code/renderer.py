# renderer.py

import cv2
import numpy as np
from utils import cartoon_filter, euclidean_distance

def overlay_hair(image, landmarks, hair_image_path):
    """
    Overlays a hair image (with alpha channel) onto the face image based on landmarks.
    """
    # Load hair image with alpha channel
    hair_img = cv2.imread(hair_image_path, cv2.IMREAD_UNCHANGED)
    if hair_img is None:
        raise Exception(f"Unable to load hair image: {hair_image_path}")

    orig_h, orig_w = image.shape[:2]

    # Calculate face width (distance between jaw corners)
    face_width = euclidean_distance(landmarks[0], landmarks[16])
    # Scale hair to match face width (with a little extra)
    hair_scale = (face_width * 1.2) / hair_img.shape[1]
    new_w = int(hair_img.shape[1] * hair_scale)
    new_h = int(hair_img.shape[0] * hair_scale)
    hair_img = cv2.resize(hair_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Determine placement: center hair on face
    face_center_x = int((landmarks[0][0] + landmarks[16][0]) / 2)
    x_offset = face_center_x - new_w // 2

    # Determine vertical placement: align hair bottom to top of eyebrows
    brow_y = min(landmarks[i][1] for i in range(17, 27))
    y_offset = brow_y - new_h
    if y_offset < 0:
        y_offset = 0

    # Prepare overlay image
    result_img = image.copy()

    # Compute overlay bounds
    x1 = max(x_offset, 0)
    x2 = min(x_offset + new_w, orig_w)
    y1 = max(y_offset, 0)
    y2 = min(y_offset + new_h, orig_h)

    hair_x1 = x1 - x_offset
    hair_x2 = hair_x1 + (x2 - x1)
    hair_y1 = y1 - y_offset
    hair_y2 = hair_y1 + (y2 - y1)

    # Blend hair onto the image using alpha channel
    alpha_hair = hair_img[hair_y1:hair_y2, hair_x1:hair_x2, 3] / 255.0
    alpha_background = 1.0 - alpha_hair
    for c in range(3):  # for each color channel
        result_img[y1:y2, x1:x2, c] = (
            alpha_hair * hair_img[hair_y1:hair_y2, hair_x1:hair_x2, c] +
            alpha_background * result_img[y1:y2, x1:x2, c]
        )

    return result_img

def apply_cartoon_filter(image):
    """
    Applies a simple cartoon-style filter to the image.
    """
    return cartoon_filter(image)
