# utils.py

import math
import cv2

def euclidean_distance(a, b):
    """Calculate Euclidean distance between two (x, y) points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def cartoon_filter(img):
    """
    Applies a cartoon effect to the input image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=9
    )
    color = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon
