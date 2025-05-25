import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

DLIB_LANDMARK_MODEL_PATH = "C:\\Users\\USER\Desktop\\Code\\Datasets\\dlib_face_68\\shape_predictor_68_face_landmarks.dat"
image_path = "C:\\Users\\USER\\Downloads\\square_face.jpg"

def capture_pic():
    camport = 0
    cam = cv2.VideoCapture(camport)
    result, image = cam.read()
    if result:
        cv2.imshow("Webcam", image)
        cv2.waitKey(0)
        cv2.destroyWindow("Weebcam")

if __name__ == "__main__":
    while True:

        if(input("") == "Y"):
            capture_pic()
            pass