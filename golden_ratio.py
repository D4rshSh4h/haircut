import cv2
import dlib
import numpy as np
import os

# --- Configuration ---
DLIB_LANDMARK_MODEL = "C:\\Users\\USER\Desktop\\Code\\Datasets\\dlib_face_68\\shape_predictor_68_face_landmarks.dat"
#MODEL_URL = f"http://dlib.net/files/{DLIB_LANDMARK_MODEL}.bz2"
SAMPLE_IMAGE_URL = "Webcam.png" # Sample male face
OUTPUT_IMAGE_NAME = "male_face_golden_ratio.jpg"
GOLDEN_RATIO_PHI = (1 + 5**0.5) / 2 # Approx 1.618
MASK_COLOR = (0, 215, 255) # BGR for Gold/Yellow
MASK_THICKNESS = 2
CROP_PADDING_FACTOR = 0.3 # Add 30% padding around the detected face bounds for cropping

# --- Helper Functions ---

def get_sample_image(image_path_or_url):
    """Loads an image from a local path or downloads from a URL."""
    if os.path.exists(image_path_or_url):
        print(f"Loading image from local path: {image_path_or_url}")
        return cv2.imread(image_path_or_url)
    else:
        print(f"Image path/URL not valid: {image_path_or_url}")
        return None

def landmarks_to_np(shape, dtype="int"):
    """Convert dlib shape object to numpy array."""
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# --- Main Processing ---

def process_face_with_golden_ratio_mask(image_path_or_url):
    """
    Detects a face, regularizes it, and draws a golden ratio mask.
    """

    # 1. Load dlib's face detector and landmark predictor
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(DLIB_LANDMARK_MODEL)
    except RuntimeError as e:
        print(f"Error loading dlib models: {e}")
        print("Make sure 'shape_predictor_68_face_landmarks.dat' is in the correct path or downloaded.")
        return

    # 2. Load the image
    original_image = get_sample_image(image_path_or_url)
    if original_image is None:
        print("Could not load the image.")
        return

    img_display = original_image.copy() # For drawing on original
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # 3. Detect faces
    faces = detector(gray, 1) # Upsample image once for better detection

    if not faces:
        print("No faces detected in the image.")
        cv2.imshow("Original Image (No Face)", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # For simplicity, process the first detected face (or largest)
    # If you want to process the largest face:
    # face = max(faces, key=lambda rect: rect.width() * rect.height())
    face_rect = faces[0]
    print(f"Face detected at: Left: {face_rect.left()} Top: {face_rect.top()} Right: {face_rect.right()} Bottom: {face_rect.bottom()}")

    # Draw detected face rectangle on original image for reference
    cv2.rectangle(img_display, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0), 2)

    # 4. Get facial landmarks
    landmarks_dlib = predictor(gray, face_rect)
    landmarks = landmarks_to_np(landmarks_dlib)

    # For drawing landmarks on original image (optional)
    # for (x, y) in landmarks:
    #     cv2.circle(img_display, (x, y), 2, (0, 0, 255), -1)

    # 5. Regularize/Crop Face
    # Define face bounding box based on landmarks for a tighter crop
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]

    # A more robust way to get top of face might be above eyebrows
    # For simplicity, we can use the min/max landmark points or dlib's detection box.
    # Let's use landmarks to define a tighter box than dlib's initial detection for cropping.
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords) # min_y is likely eyebrow top, max_y is chin

    # Estimate forehead top: extend upwards from eyebrows.
    # Eyebrows are roughly landmarks 17-21 (left) and 22-26 (right)
    eyebrow_y_avg = np.mean(landmarks[17:27, 1])
    chin_y = landmarks[8, 1] # Landmark 8 is the chin tip
    face_height_landmarks = chin_y - eyebrow_y_avg
    
    # Estimate forehead top to be a certain proportion above eyebrows
    # This is an approximation. A true hairline is hard to get from 68 landmarks.
    est_forehead_top_y = int(eyebrow_y_avg - face_height_landmarks * 0.4) # Adjust 0.4 factor as needed
    min_y_crop = max(0, est_forehead_top_y) # Ensure it's not negative
    max_y_crop = int(chin_y)

    # Crop boundaries with padding
    pad_w = int((max_x - min_x) * CROP_PADDING_FACTOR / 2)
    pad_h = int((max_y_crop - min_y_crop) * CROP_PADDING_FACTOR / 2)

    crop_x1 = max(0, min_x - pad_w)
    crop_y1 = max(0, min_y_crop - pad_h)
    crop_x2 = min(original_image.shape[1], max_x + pad_w)
    crop_y2 = min(original_image.shape[0], max_y_crop + pad_h)
    
    # Ensure crop coordinates are valid
    if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
        print("Calculated crop dimensions are invalid. Using dlib's detected face box.")
        crop_x1, crop_y1, crop_x2, crop_y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()


    cropped_face = original_image[crop_y1:crop_y2, crop_x1:crop_x2]

    if cropped_face.size == 0:
        print("Cropped face is empty. Something went wrong.")
        return

    # Adjust landmarks to be relative to the cropped image
    # landmarks_cropped = landmarks.copy()
    # landmarks_cropped[:, 0] -= crop_x1
    # landmarks_cropped[:, 1] -= crop_y1

    # 6. Draw Golden Ratio Mask on the Cropped Face
    # We will draw the grid directly on the cropped_face using its dimensions
    h_crop, w_crop = cropped_face.shape[:2]
    
    # Make a copy to draw the mask on
    face_with_mask = cropped_face.copy()

    # --- Horizontal Lines ---
    # Divide height by phi, then the larger remaining segment by phi again.
    # Line 1 (from top)
    y1 = int(h_crop / GOLDEN_RATIO_PHI)
    cv2.line(face_with_mask, (0, y1), (w_crop, y1), MASK_COLOR, MASK_THICKNESS)
    
    # Line 2 (from bottom, mirrors the top division)
    # This divides the larger segment (bottom one) by phi again from its end
    y2 = int(h_crop - (h_crop / GOLDEN_RATIO_PHI)) # Equivalent to h_crop * (1 - 1/PHI) = h_crop / PHI^2
    cv2.line(face_with_mask, (0, y2), (w_crop, y2), MASK_COLOR, MASK_THICKNESS)

    # --- Vertical Lines ---
    # Divide width by phi, then the larger remaining segment by phi again.
    # Line 1 (from left)
    x1 = int(w_crop / GOLDEN_RATIO_PHI)
    cv2.line(face_with_mask, (x1, 0), (x1, h_crop), MASK_COLOR, MASK_THICKNESS)

    # Line 2 (from right)
    x2 = int(w_crop - (w_crop / GOLDEN_RATIO_PHI))
    cv2.line(face_with_mask, (x2, 0), (x2, h_crop), MASK_COLOR, MASK_THICKNESS)

    # Also draw the outer bounding box of the crop
    cv2.rectangle(face_with_mask, (0,0), (w_crop-1, h_crop-1), MASK_COLOR, MASK_THICKNESS)


    # 7. Display results
    cv2.imshow("Original Image with Detection", img_display)
    cv2.imshow("Cropped Face", cropped_face)
    cv2.imshow("Face with Golden Ratio Mask", face_with_mask)
    
    # Save the output
    cv2.imwrite(OUTPUT_IMAGE_NAME, face_with_mask)
    print(f"Output image saved as {OUTPUT_IMAGE_NAME}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- Run the script ---
if __name__ == "__main__":
    # You can replace SAMPLE_IMAGE_URL with a local path to your image
    # e.g., "path/to/your/male_face.jpg"
    image_source = SAMPLE_IMAGE_URL 
    # image_source = "my_local_face.jpg" # if you have a local file
    
    process_face_with_golden_ratio_mask(image_source)