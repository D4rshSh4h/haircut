import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
# Path to Dlib's pre-trained facial landmark predictor
DLIB_LANDMARK_MODEL_PATH = "C:\\Users\\USER\Desktop\\Code\\Datasets\\dlib_face_68\\shape_predictor_68_face_landmarks.dat" #UPDATE THIS PATH

# --- Helper Functions ---
def landmarks_to_np(landmarks, dtype="int"):
    """Convert Dlib's full_object_detection to a NumPy array."""
    coords = np.zeros((landmarks.num_parts, 2), dtype=dtype)
    for i in range(0, landmarks.num_parts):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    return coords

def visualize_landmarks(image, landmarks, face_rect=None, title="Facial Landmarks"):
    """Draw landmarks and face rectangle on the image."""
    img_display = image.copy()
    if face_rect:
        (x, y, w, h) = (face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height())
        cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y) in landmarks:
        cv2.circle(img_display, (x, y), 2, (0, 0, 255), -1)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# --- Core Face Analysis Functions ---

class FaceAnalyzer:
    def __init__(self, landmark_model_path):
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(landmark_model_path)
        except RuntimeError as e:
            print(f"Error loading Dlib model: {e}")
            print(f"Ensure '{landmark_model_path}' exists and is accessible.")
            raise

    def analyze_face_from_path(self, image_path):
        """Loads an image, detects face, landmarks, and analyzes shape."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image from {image_path}")
                return None, None, None, None
        except Exception as e:
            print(f"Error reading image file {image_path}: {e}")
            return None, None, None, None

        return self.analyze_face_from_image(image)

    def analyze_face_from_image(self, image):
        """Detects face, landmarks, and analyzes shape from an image array."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1) # Upsample image 1 time to find smaller faces

        if len(faces) == 0:
            print("No faces detected.")
            return image, None, None, "No face detected"

        # For simplicity, we'll process the first detected face
        # In a real system, you might choose the largest face or allow user selection
        face_rect = faces[0]
        landmarks_dlib = self.predictor(gray, face_rect)
        landmarks_np = landmarks_to_np(landmarks_dlib)

        face_shape = self._determine_face_shape(landmarks_np)
        # You can add golden ratio and other feature calculations here

        return image, face_rect, landmarks_np, face_shape

    def _determine_face_shape(self, landmarks):
        """
        Determines face shape based on landmark geometry.
        This is a rule-based heuristic and can be improved with more sophisticated methods.
        Dlib 68 landmarks indices:
        - Jawline: 0-16
        - Left eyebrow: 17-21
        - Right eyebrow: 22-26
        - Nose bridge: 27-30
        - Nose tip: 30
        - Lower nose: 31-35
        - Left eye: 36-41
        - Right eye: 42-47
        - Outer mouth: 48, 54
        - Inner mouth: 60, 64
        """

        # --- Calculate key facial dimensions ---
        # Face width (approximated at cheekbones or widest part of jaw/temples)
        # Using points 1 and 15 as rough cheekbone/temple width indicators
        face_width_cheek = np.linalg.norm(landmarks[1] - landmarks[15])

        # Using jaw points for another width measure
        jaw_points = landmarks[0:17]
        jaw_width = np.linalg.norm(jaw_points[2] - jaw_points[14]) # A bit narrower than full jaw

        # Overall face width could be max of several measures
        # For simplicity, let's use a robust one: distance between points 0 and 16 (outer jaw)
        overall_face_width = np.linalg.norm(landmarks[0] - landmarks[16])


        # Face length (chin to approx. hairline/top of forehead)
        # Point 8 is chin. For top, we can use mid-point above eyebrows or nose bridge.
        # Let's use point 27 (top of nose bridge) as a proxy for "mid-forehead" area.
        # A better approach might involve estimating hairline.
        chin_point = landmarks[8]
        forehead_approx_point_y = min(landmarks[19][1], landmarks[24][1]) # Top of eyebrows
        # A more stable "top of face" could be a point projected upwards from landmark 27
        # or an estimation based on eye-to-chin distance.
        # For now, let's use a simple chin-to-eyebrow-top distance as "face height"
        # or chin (8) to top of nose bridge (27) and scale it slightly.
        # A more common one: chin (8) to a point slightly above mid-eyebrows.
        # Let's try chin to a point just above landmark 27 (bridge of nose)
        top_of_forehead_y_estimate = landmarks[27][1] - (landmarks[8][1] - landmarks[27][1]) * 0.2 # Heuristic
        face_length = landmarks[8][1] - top_of_forehead_y_estimate
        # Alternative face length: chin (8) to highest point of eyebrows (19 or 24)
        # face_length = np.linalg.norm(landmarks[8] - landmarks[27]) # Chin to nose bridge
        # face_length_alternative = landmarks[8][1] - min(landmarks[19][1], landmarks[24][1]) # Chin to top of eyebrows

        # Forehead width (between temples - points 0 and 16 can be too wide)
        # Using outer points of eyebrows (17 and 26)
        forehead_width = np.linalg.norm(landmarks[17] - landmarks[26])

        # Jawline width (points 4 and 12 or 5 and 11)
        jawline_width = np.linalg.norm(landmarks[4] - landmarks[12])


        # --- Ratios for classification ---
        # These thresholds are heuristic and need tuning!
        if overall_face_width == 0: return "Unknown (division by zero)" # Should not happen if face detected
        length_to_width_ratio = face_length / overall_face_width
        forehead_to_jaw_ratio = forehead_width / jawline_width
        cheek_to_jaw_ratio = face_width_cheek / jawline_width # Using the cheek measure here

        # --- Classification Logic (Simplified Rule-Based) ---
        shape = "Unknown"

        # print(f"  Debug Ratios: L/W={length_to_width_ratio:.2f}, Forehead/Jaw={forehead_to_jaw_ratio:.2f}, Cheek/Jaw={cheek_to_jaw_ratio:.2f}")
        # print(f"  Debug Dims: Length={face_length:.2f}, Width={overall_face_width:.2f}, ForeheadW={forehead_width:.2f}, JawW={jawline_width:.2f}")


        if length_to_width_ratio > 1.25: # Likely longer than wide
            if forehead_width > jawline_width * 0.95 and jawline_width > overall_face_width * 0.7: # Forehead and jaw somewhat similar, not too tapered
                shape = "Oblong"
            elif forehead_width > jawline_width and jawline_width < overall_face_width * 0.85 : # Forehead wider, jaw tapers
                shape = "Heart" # Could also be inverted triangle
            elif face_width_cheek > forehead_width and face_width_cheek > jawline_width:
                shape = "Diamond" # Widest at cheeks
            else:
                shape = "Oval" # Default for longer faces if other conditions not met
        elif length_to_width_ratio < 0.95 : # Wider than long or roughly equal, suggests round or square
            if abs(forehead_width - jawline_width) < (overall_face_width * 0.15) and abs(overall_face_width - jawline_width) < (overall_face_width * 0.15):
                # Jaw is strong and similar width to forehead/cheeks
                shape = "Square"
            else:
                shape = "Round"
        else: # Proportions are somewhat balanced (length_to_width_ratio between 0.95 and 1.25)
            if forehead_width > jawline_width and jawline_width < overall_face_width * 0.85:
                shape = "Heart" # Tapering jaw
            elif face_width_cheek > forehead_width and face_width_cheek > jawline_width:
                shape = "Diamond"
            elif abs(forehead_width - jawline_width) < (overall_face_width * 0.15) and jawline_width > overall_face_width * 0.85 :
                # Jaw is relatively strong and similar width to forehead
                shape = "Square" # Can also be square if L/W is ~1
            elif forehead_width > jawline_width * 0.9 and jawline_width > overall_face_width * 0.75: # Rounded jaw, forehead slightly wider or equal
                shape = "Oval"
            else:
                shape = "Round" # Default for balanced if other specific conditions aren't met

        # Refinements:
        # Triangle: jawline significantly wider than forehead.
        if jawline_width > forehead_width * 1.15:
            shape = "Triangle"

        # Override Oval if L/W is very close to 1 and other features point to round/square
        if 0.95 <= length_to_width_ratio <= 1.05:
            if shape == "Oval": # Re-evaluate if Oval was chosen for balanced L/W
                if abs(forehead_width - jawline_width) < (overall_face_width * 0.15) and abs(overall_face_width - jawline_width) < (overall_face_width * 0.15):
                    shape = "Square"
                else:
                    shape = "Round"

        return shape

    def get_golden_ratio_metrics(self, landmarks):
        """
        Calculates various facial proportions relative to the golden ratio (approx 1.618).
        Returns a dictionary of metrics and their "closeness" to the golden ratio.
        This is for informational purposes; "beauty" is subjective.
        """
        metrics = {}
        golden_ratio = 1.618

        # 1. Face Length / Face Width
        # Using points 0 and 16 for width, and 8 (chin) to approx forehead top for length
        face_width = np.linalg.norm(landmarks[0] - landmarks[16])
        # For length, estimate top of forehead: project upwards from landmark 27 (nose bridge)
        # This is a very rough estimate.
        estimated_forehead_top_y = landmarks[27][1] - (landmarks[8][1] - landmarks[27][1]) * 0.3 # More aggressive projection
        face_length = landmarks[8][1] - estimated_forehead_top_y
        if face_width > 0:
            val = face_length / face_width
            metrics["Face_Length_to_Width"] = {"value": val, "ideal": golden_ratio, "diff_percentage": abs(val - golden_ratio) / golden_ratio * 100}

        # 2. Nose Length / Nose Width
        # Nose length: point 27 (top of bridge) to 33 (bottom center)
        nose_length = np.linalg.norm(landmarks[27] - landmarks[33])
        # Nose width: point 31 to 35 (nostril edges)
        nose_width = np.linalg.norm(landmarks[31] - landmarks[35])
        if nose_width > 0:
            val = nose_length / nose_width
            metrics["Nose_Length_to_Width"] = {"value": val, "ideal": golden_ratio, "diff_percentage": abs(val - golden_ratio) / golden_ratio * 100}


        # 3. Mouth Width / Nose Width
        # Mouth width: point 48 to 54 (corners of mouth)
        mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
        if nose_width > 0:
            val = mouth_width / nose_width
            metrics["Mouth_Width_to_Nose_Width"] = {"value": val, "ideal": golden_ratio, "diff_percentage": abs(val - golden_ratio) / golden_ratio * 100}

        # 4. Distance between pupils / Mouth width
        # Pupil centers (approx): mid-points of eye landmarks
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)
        inter_pupillary_distance = np.linalg.norm(left_eye_center - right_eye_center)
        if mouth_width > 0:
            val = inter_pupillary_distance / mouth_width
            # This ratio is sometimes compared to 1, or other ideals, not always GR.
            # For consistency, we'll use GR, but note its applicability varies.
            metrics["Interpupillary_to_Mouth_Width"] = {"value": val, "ideal": 1.0, "diff_percentage": abs(val - 1.0) / 1.0 * 100} # Ideal is often 1 here

        # Add more ratios as needed:
        # - Forehead height / Nose length / Lower face height (chin to nose bottom)
        # - Eye width / Interocular distance

        return metrics

    def get_other_features(self, landmarks):
        """Extract other potentially useful features."""
        features = {}

        # Forehead Height (approximate: top of eyebrows to estimated hairline)
        # Difficult to do accurately without hairline detection.
        # Using distance from eyebrow top (avg of 19, 24) to nose bridge (27) as a proxy.
        eyebrow_top_y = (landmarks[19][1] + landmarks[24][1]) / 2
        nose_bridge_y = landmarks[27][1]
        # We can compare this to face_length or another feature
        # For now, let's just provide the raw relative distance
        # A simple category:
        face_height_chin_to_nose_bridge = landmarks[8][1] - landmarks[27][1]
        forehead_proxy_height = eyebrow_top_y - (landmarks[27][1] - face_height_chin_to_nose_bridge * 0.2) # estimate top of forehead
        # Relative forehead height
        if face_height_chin_to_nose_bridge > 0:
            relative_forehead_height = (eyebrow_top_y - nose_bridge_y) / face_height_chin_to_nose_bridge
            if relative_forehead_height < 0.4: features["Forehead_Relative_Height"] = "Low"
            elif relative_forehead_height > 0.6: features["Forehead_Relative_Height"] = "High"
            else: features["Forehead_Relative_Height"] = "Average"

        # Jawline Definition (angle from point 4/5 to 8 to 11/12)
        p5 = landmarks[5]
        p8_chin = landmarks[8]
        p11 = landmarks[11]
        # Create vectors
        v1 = p5 - p8_chin
        v2 = p11 - p8_chin
        # Calculate angle (cosine rule)
        jaw_angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        jaw_angle_deg = np.degrees(jaw_angle_rad)
        features["Jaw_Angle_Degrees"] = jaw_angle_deg
        if jaw_angle_deg < 110: features["Jawline_Definition"] = "Strong/Angular"
        elif jaw_angle_deg > 130: features["Jawline_Definition"] = "Soft/Rounded"
        else: features["Jawline_Definition"] = "Defined"

        # Cheekbone Prominence (width at points 1-2 vs 14-15 relative to jaw or forehead)
        cheekbone_width_upper = np.linalg.norm(landmarks[2] - landmarks[14]) # A bit higher on cheek
        jawline_width = np.linalg.norm(landmarks[4] - landmarks[12])
        if jawline_width > 0:
            cheek_to_jaw_width_ratio = cheekbone_width_upper / jawline_width
            if cheek_to_jaw_width_ratio > 1.1: features["Cheekbone_Prominence"] = "High"
            elif cheek_to_jaw_width_ratio < 0.95: features["Cheekbone_Prominence"] = "Subtle"
            else: features["Cheekbone_Prominence"] = "Average"

        return features


# --- Main Execution ---
if __name__ == "__main__":
    # Create an instance of the analyzer
    try:
        analyzer = FaceAnalyzer(landmark_model_path=DLIB_LANDMARK_MODEL_PATH)
    except Exception:
        print("Failed to initialize FaceAnalyzer. Exiting.")
        exit()

    # --- Example Usage ---
    # Replace 'path/to/your/face_image.jpg' with an actual image path
    # Try images with different face shapes to test the classification.
    # For example:
    # - Emma Stone (Oval/Round)
    # - Olivia Wilde (Square)
    # - Angelina Jolie (Square/Heart)
    # - Reese Witherspoon (Heart)
    # - Sarah Jessica Parker (Oblong)
    # - Ginnifer Goodwin (Round)
    # - Lucy Liu (Diamond - though Dlib might struggle with precise diamond features)

    image_path = "C:\\Users\\USER\\Downloads\\square_face.jpg" # <--- IMPORTANT: SET YOUR IMAGE PATH HERE

    print(f"Analyzing image: {image_path}")
    image, face_rect, landmarks, face_shape = analyzer.analyze_face_from_path(image_path)

    if landmarks is not None:
        print(f"\n--- Analysis Results ---")
        print(f"Detected Face Shape: {face_shape}")

        print("\n--- Golden Ratio Metrics (approximate) ---")
        golden_metrics = analyzer.get_golden_ratio_metrics(landmarks)
        for key, data in golden_metrics.items():
            print(f"  {key}: {data['value']:.2f} (Ideal: {data['ideal']:.2f}, Diff: {data['diff_percentage']:.1f}%)")

        print("\n--- Other Facial Features ---")
        other_feats = analyzer.get_other_features(landmarks)
        for key, value in other_feats.items():
            print(f"  {key}: {value}")

        visualize_landmarks(image, landmarks, face_rect, title=f"Detected Face Shape: {face_shape}")
    else:
        if image is not None: # Image loaded but no face found
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Image Loaded - No Face Detected")
            plt.axis('off')
            plt.show()
        else:
            print(f"Could not process image: {image_path}")