import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

DLIB_LANDMARK_MODEL_PATH = "C:\\Users\\USER\Desktop\\Code\\Datasets\\dlib_face_68\\shape_predictor_68_face_landmarks.dat"
image_path = "C:\\Users\\USER\\Downloads\\square_face.jpg"
image_name = "Webcam"

def capture_pic():
    camport = 0
    cam = cv2.VideoCapture(camport)
    result, image = cam.read()
    if result:
        cv2.imshow(image_name, image)
        cv2.imwrite(image_name+".png", image)
        cv2.waitKey(0)
        cv2.destroyWindow(image_name)

def get_image():
    pass


if __name__ == "__main__":
    capture_pic()