# main.py

import cv2
import argparse
from face_analysis import detect_landmarks, classify_face_shape
from recommender import recommend_hair
from renderer import overlay_hair, apply_cartoon_filter

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Face Shape-Based Hairstyle Recommender and Styler")
    parser.add_argument("--image", required=False, help="Path to input face image")
    parser.add_argument("--output", required=False, help="Path to save the output image")
    args = parser.parse_args()

    # Filepath configuration
    filepath = args.image if args.image else "assets/input_face.jpg"
    output_path = args.output if args.output else "output.png"

    # Load the input image
    image = cv2.imread(filepath)
    if image is None:
        print(f"Error: Unable to load image at {filepath}")
        return

    # Detect facial landmarks and determine face shape
    try:
        landmarks = detect_landmarks(image)
    except Exception as e:
        print(f"Face detection failed: {e}")
        return

    face_shape = classify_face_shape(landmarks)
    print(f"Detected face shape: {face_shape}")

    # Recommend a hairstyle based on face shape
    hair_path = recommend_hair(face_shape)
    if hair_path is None:
        print(f"No hairstyle recommendation available for face shape: {face_shape}")
        return
    print(f"Selected hairstyle overlay: {hair_path}")

    # Overlay the hairstyle onto the face image
    stylized_image = overlay_hair(image, landmarks, hair_path)

    # Apply a cartoon filter to stylize the image
    cartoon_image = apply_cartoon_filter(stylized_image)

    # Save the final result
    cv2.imwrite(output_path, cartoon_image)
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main()
