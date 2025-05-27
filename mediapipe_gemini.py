import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

# --- Landmark Index Constants (Approximations or Well-Known Indices) ---
# These are simplified for direct use; for robust use with all connections,
# MediaPipe's own mp_face_mesh.FACEMESH_... connection sets are preferred for drawing.
def get_unique_indices_from_connections(connections_set):
    indices = set()
    if connections_set: # Check if the set is not None
        for conn in connections_set:
            indices.add(conn[0])
            indices.add(conn[1])
    return sorted(list(indices)) if indices else []

FACEMESH_LIPS = get_unique_indices_from_connections(getattr(mp_face_mesh, 'FACEMESH_LIPS', None))
FACEMESH_LEFT_EYE = get_unique_indices_from_connections(getattr(mp_face_mesh, 'FACEMESH_LEFT_EYE', None))
FACEMESH_LEFT_EYEBROW = get_unique_indices_from_connections(getattr(mp_face_mesh, 'FACEMESH_LEFT_EYEBROW', None))
FACEMESH_LEFT_IRIS = get_unique_indices_from_connections(getattr(mp_face_mesh, 'FACEMESH_LEFT_IRIS', None))
FACEMESH_RIGHT_EYE = get_unique_indices_from_connections(getattr(mp_face_mesh, 'FACEMESH_RIGHT_EYE', None))
FACEMESH_RIGHT_EYEBROW = get_unique_indices_from_connections(getattr(mp_face_mesh, 'FACEMESH_RIGHT_EYEBROW', None))
FACEMESH_RIGHT_IRIS = get_unique_indices_from_connections(getattr(mp_face_mesh, 'FACEMESH_RIGHT_IRIS', None))
FACEMESH_FACE_OVAL = get_unique_indices_from_connections(getattr(mp_face_mesh, 'FACEMESH_FACE_OVAL', None))

FACEMESH_NOSE_TIP = 1
FACEMESH_CHIN = 152
FACEMESH_FOREHEAD_CENTER = 10
FACEMESH_LEFT_FACE_EDGE = 234
FACEMESH_RIGHT_FACE_EDGE = 454
FACEMESH_NOSE_BRIDGE_TOP = 168
FACEMESH_NOSE_BOTTOM_CENTER = 2
FACEMESH_NOSE_LEFT_ALA_APPROX = 240
FACEMESH_NOSE_RIGHT_ALA_APPROX = 460

# --- Helper Functions ---
def _mediapipe_landmarks_to_np(face_landmarks_proto, image_width, image_height, dtype="int"):
    num_landmarks = len(face_landmarks_proto.landmark)
    coords = np.zeros((num_landmarks, 2), dtype=dtype)
    for i, landmark in enumerate(face_landmarks_proto.landmark):
        coords[i] = (int(landmark.x * image_width), int(landmark.y * image_height))
    return coords

def create_full_head_mask_image(original_image, person_segmentation_mask):
    if person_segmentation_mask is None:
        print("Warning: Person segmentation mask is None. Returning original image for full head mask.")
        return original_image.copy()
    
    mask_3channel = person_segmentation_mask
    if person_segmentation_mask.ndim == 2:
        mask_3channel = np.stack((person_segmentation_mask,) * 3, axis=-1)
    elif person_segmentation_mask.shape[-1] == 1:
        mask_3channel = np.concatenate((person_segmentation_mask,) * 3, axis=-1)
    
    if mask_3channel.shape[:2] != original_image.shape[:2]: # Resize mask if needed
        print(f"Resizing segmentation mask from {mask_3channel.shape[:2]} to {original_image.shape[:2]}")
        mask_3channel = cv2.resize(mask_3channel, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        if mask_3channel.ndim == 2: # cv2.resize might output 2D if input was 2D
             mask_3channel = np.stack((mask_3channel,) * 3, axis=-1)


    black_background = np.zeros_like(original_image, dtype=np.uint8)
    try:
        # Ensure mask is binary {0, 1} if it's not already
        if mask_3channel.max() > 1 and np.issubdtype(mask_3channel.dtype, np.number): # Check if it's not already {0,1}
            mask_3channel = (mask_3channel > 0.5).astype(np.uint8) # Assuming a threshold was applied earlier

        masked_image = np.where(mask_3channel == 1, original_image, black_background)
    except ValueError as e:
        print(f"Error during np.where for masking: {e}. Shapes: mask={mask_3channel.shape}, image={original_image.shape}")
        return original_image.copy()
    return masked_image

def draw_full_facemesh_on_image(image, face_mesh_results):
    annotated_image = image.copy()
    if face_mesh_results and face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            if hasattr(mp_face_mesh, 'FACEMESH_TESSELATION') and mp_face_mesh.FACEMESH_TESSELATION:
                mp_drawing.draw_landmarks(
                    image=annotated_image, landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            if hasattr(mp_face_mesh, 'FACEMESH_CONTOURS') and mp_face_mesh.FACEMESH_CONTOURS:
                mp_drawing.draw_landmarks(
                    image=annotated_image, landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            if hasattr(mp_face_mesh, 'FACEMESH_IRISES') and mp_face_mesh.FACEMESH_IRISES:
                 mp_drawing.draw_landmarks(
                    image=annotated_image, landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None, 
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
    return annotated_image

def visualize_analysis_results(original_image, full_head_masked_image, 
                               image_with_full_facemesh, landmarks_np, face_bbox=None, 
                               title_prefix="Facial Analysis"):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    if full_head_masked_image is not None: plt.imshow(cv2.cvtColor(full_head_masked_image, cv2.COLOR_BGR2RGB))
    else: plt.text(0.5, 0.5, "Masked image NA", ha='center', va='center')
    plt.title("Full Head Mask (Hair Included)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    if image_with_full_facemesh is not None: plt.imshow(cv2.cvtColor(image_with_full_facemesh, cv2.COLOR_BGR2RGB))
    else: plt.text(0.5, 0.5, "Full facemesh NA", ha='center', va='center')
    plt.title("Full Face Mesh")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    img_display_bbox_title = original_image.copy()
    if face_bbox:
        (x, y, w, h) = face_bbox
        cv2.rectangle(img_display_bbox_title, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if landmarks_np is not None:
        key_indices_to_draw = [FACEMESH_CHIN, FACEMESH_FOREHEAD_CENTER]
        for idx in key_indices_to_draw:
            if idx < len(landmarks_np): cv2.circle(img_display_bbox_title, tuple(landmarks_np[idx]), 3, (0, 0, 255), -1)
    plt.imshow(cv2.cvtColor(img_display_bbox_title, cv2.COLOR_BGR2RGB))
    plt.title(f"{title_prefix} (with BBox)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- Core Face Analysis Functions ---
class FaceAnalyzer:
    def __init__(self, landmark_model_path=None):
        try:
            self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
            print("MediaPipe models initialized.")
        except Exception as e: print(f"Error initializing MediaPipe models: {e}"); raise

    def analyze_face_from_path(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None: return image_path, None, None, None, None, None, "Image load error"
        except Exception as e: return image_path, None, None, None, None, None, f"Image read error: {e}"
        return self.analyze_face_from_image(image)

    def analyze_face_from_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = image_rgb.shape
        mask_bin, landmarks_np, bbox, shape, full_head_mask_img, mesh_res = None, None, None, "Unknown", image.copy(), None

        seg_res = self.selfie_segmentation.process(image_rgb)
        if seg_res.segmentation_mask is not None:
            mask_bin = (seg_res.segmentation_mask > 0.5).astype(np.uint8)
            full_head_mask_img = create_full_head_mask_image(image, mask_bin)
        else: print("Warning: Selfie segmentation failed.")

        mesh_res = self.face_mesh.process(image_rgb)
        if mesh_res.multi_face_landmarks:
            landmarks_proto = mesh_res.multi_face_landmarks[0]
            landmarks_np = _mediapipe_landmarks_to_np(landmarks_proto, w_img, h_img)
            x_coords, y_coords = landmarks_np[:, 0], landmarks_np[:, 1]
            min_x, max_x, min_y, max_y = np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)
            bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
            shape = self._determine_face_shape_mp(landmarks_np, bbox)
        else: 
            print("No face detected by FaceMesh.")
            return image, full_head_mask_img, None, None, mask_bin, mesh_res, "No face by Mesh"
        return image, full_head_mask_img, bbox, landmarks_np, mask_bin, mesh_res, shape

    def _get_coords(self, landmarks_np, indices):
        if landmarks_np is None: return None
        indices = [indices] if not isinstance(indices, list) else indices
        pts = [landmarks_np[i] for i in indices if i < len(landmarks_np)]
        return np.array(pts) if pts else None
    
    def _determine_face_shape_mp(self, landmarks_np, face_bbox):
        if face_bbox is None or landmarks_np is None: return "Unknown"
        x_bbox, y_bbox, w_bbox, h_bbox = face_bbox
        if w_bbox == 0 or h_bbox == 0: return "Unknown (zero dim bbox)"
        aspect_ratio = h_bbox / w_bbox
        shape = "Oval"
        if aspect_ratio > 1.35: shape = "Oblong/Oval"
        elif aspect_ratio < 0.95: shape = "Round/Square"
        
        try:
            # Using approx temple points 103 (L), 332 (R) for forehead width
            # Using approx jaw angle points 172 (L), 397 (R) for jaw width
            forehead_l, forehead_r = self._get_coords(landmarks_np, 103), self._get_coords(landmarks_np, 332)
            jaw_l, jaw_r = self._get_coords(landmarks_np, 172), self._get_coords(landmarks_np, 397)
            if not all([forehead_l is not None, forehead_r is not None, jaw_l is not None, jaw_r is not None]): return shape
            
            forehead_w = np.linalg.norm(forehead_l[0] - forehead_r[0])
            jaw_w = np.linalg.norm(jaw_l[0] - jaw_r[0])

            if forehead_w > 0 and jaw_w > 0:
                ratio = forehead_w / jaw_w
                if aspect_ratio > 1.15: # Taller faces
                    if ratio < 0.9: shape = "Triangle/Pear"
                    elif ratio > 1.1 and jaw_w < forehead_w * 0.85: shape = "Heart"
                elif shape == "Round/Square":
                    if abs(forehead_w - jaw_w) < (w_bbox * 0.15): shape = "Square"
                    else: shape = "Round"
        except: pass
        return shape

    def get_golden_ratio_metrics(self, landmarks_np):
        metrics = {}
        phi = 1.618
        if landmarks_np is None: return metrics
        try:
            face_w = np.linalg.norm(self._get_coords(landmarks_np, FACEMESH_LEFT_FACE_EDGE)[0] - self._get_coords(landmarks_np, FACEMESH_RIGHT_FACE_EDGE)[0])
            face_l = np.linalg.norm(self._get_coords(landmarks_np, FACEMESH_CHIN)[0] - self._get_coords(landmarks_np, FACEMESH_FOREHEAD_CENTER)[0])
            if face_w > 0: metrics["Face_Length_to_Width"] = {"value": face_l / face_w, "ideal": phi, "diff_percentage": abs(face_l/face_w - phi) / phi * 100}

            nose_l = np.linalg.norm(self._get_coords(landmarks_np, FACEMESH_NOSE_BRIDGE_TOP)[0] - self._get_coords(landmarks_np, FACEMESH_NOSE_BOTTOM_CENTER)[0])
            nose_w = np.linalg.norm(self._get_coords(landmarks_np, FACEMESH_NOSE_LEFT_ALA_APPROX)[0] - self._get_coords(landmarks_np, FACEMESH_NOSE_RIGHT_ALA_APPROX)[0])
            if nose_w > 0: metrics["Nose_Length_to_Width"] = {"value": nose_l / nose_w, "ideal": phi, "diff_percentage": abs(nose_l/nose_w - phi) / phi * 100}
            
            mouth_w = np.linalg.norm(self._get_coords(landmarks_np, 61)[0] - self._get_coords(landmarks_np, 291)[0]) # Mouth corners
            if nose_w > 0: metrics["Mouth_Width_to_Nose_Width"] = {"value": mouth_w / nose_w, "ideal": phi, "diff_percentage": abs(mouth_w/nose_w - phi) / phi * 100}

            l_iris_pts, r_iris_pts = self._get_coords(landmarks_np, FACEMESH_LEFT_IRIS), self._get_coords(landmarks_np, FACEMESH_RIGHT_IRIS)
            if l_iris_pts is not None and r_iris_pts is not None and len(l_iris_pts)>0 and len(r_iris_pts)>0 :
                pupil_dist = np.linalg.norm(np.mean(l_iris_pts, axis=0) - np.mean(r_iris_pts, axis=0))
                if mouth_w > 0: metrics["Interpupillary_to_Mouth_Width"] = {"value": pupil_dist / mouth_w, "ideal": 1.0, "diff_percentage": abs(pupil_dist/mouth_w - 1.0) / 1.0 * 100}
        except: pass
        return metrics

    def get_other_features(self, landmarks_np):
        features = {}
        if landmarks_np is None: return features
        try:
            l_eb_pts, r_eb_pts = self._get_coords(landmarks_np, FACEMESH_LEFT_EYEBROW), self._get_coords(landmarks_np, FACEMESH_RIGHT_EYEBROW)
            if l_eb_pts is not None and r_eb_pts is not None:
                eb_top_y = (np.min(l_eb_pts[:,1]) + np.min(r_eb_pts[:,1])) / 2
                forehead_c_pt = self._get_coords(landmarks_np, FACEMESH_FOREHEAD_CENTER)
                chin_pt = self._get_coords(landmarks_np, FACEMESH_CHIN)
                if forehead_c_pt is not None and chin_pt is not None:
                    fh_h_abs = abs(eb_top_y - forehead_c_pt[0][1])
                    face_l_y = abs(chin_pt[0][1] - forehead_c_pt[0][1])
                    if face_l_y > 0:
                        rel_fh_h = fh_h_abs / face_l_y
                        if rel_fh_h < 0.22: features["Forehead_Relative_Height"] = "Low"
                        elif rel_fh_h > 0.35: features["Forehead_Relative_Height"] = "High"
                        else: features["Forehead_Relative_Height"] = "Average"

            jaw_l_pt, chin_j_pt, jaw_r_pt = self._get_coords(landmarks_np, 172), self._get_coords(landmarks_np, FACEMESH_CHIN), self._get_coords(landmarks_np, 397)
            if all([jaw_l_pt is not None, chin_j_pt is not None, jaw_r_pt is not None]):
                v1, v2 = jaw_l_pt[0] - chin_j_pt[0], jaw_r_pt[0] - chin_j_pt[0]
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
                    features["Jaw_Angle_Degrees"] = angle
                    if angle < 125: features["Jawline_Definition"] = "Strong/Angular"
                    elif angle > 150: features["Jawline_Definition"] = "Soft/Rounded"
                    else: features["Jawline_Definition"] = "Defined"

            cheek_l_pt, cheek_r_pt = self._get_coords(landmarks_np, 103), self._get_coords(landmarks_np, 332) # Upper cheek width
            if cheek_l_pt is not None and cheek_r_pt is not None and jaw_l_pt is not None and jaw_r_pt is not None:
                cheek_w = np.linalg.norm(cheek_l_pt[0] - cheek_r_pt[0])
                jaw_w_ratio = np.linalg.norm(jaw_l_pt[0] - jaw_r_pt[0])
                if jaw_w_ratio > 0:
                    ratio = cheek_w / jaw_w_ratio
                    if ratio > 1.0: features["Cheekbone_Prominence"] = "High"
                    elif ratio < 0.95: features["Cheekbone_Prominence"] = "Subtle"
                    else: features["Cheekbone_Prominence"] = "Average"
        except: pass
        return features

# --- Main Execution ---
if __name__ == "__main__":
    IMAGE_PATH = "Webcam.png" # <--- IMPORTANT: SET YOUR IMAGE PATH HERE
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image not found at '{IMAGE_PATH}'. Please provide a valid path.")
        exit()

    try: analyzer = FaceAnalyzer()
    except Exception as e: print(f"Analyzer init failed: {e}"); exit()
        
    print(f"Analyzing image: {IMAGE_PATH}")
    orig_img, head_mask_img, bbox, landmarks, _, mesh_raw, shape = analyzer.analyze_face_from_path(IMAGE_PATH)

    full_facemesh_img = None
    if orig_img is not None and isinstance(orig_img, np.ndarray):
        if mesh_raw: full_facemesh_img = draw_full_facemesh_on_image(orig_img, mesh_raw)

        if landmarks is not None and bbox is not None:
            print(f"\n--- Analysis Results ---")
            print(f"Detected Face Shape (Simplified): {shape}")
            print("\n--- Golden Ratio Metrics ---"); [print(f"  {k}: {v.get('value',0):.2f} (Ideal: {v.get('ideal',0):.2f}, Diff: {v.get('diff_percentage',0):.1f}%)") for k,v in analyzer.get_golden_ratio_metrics(landmarks).items()]
            print("\n--- Other Facial Features ---"); [print(f"  {k}: {v}") for k,v in analyzer.get_other_features(landmarks).items()]
            visualize_analysis_results(orig_img, head_mask_img, full_facemesh_img, landmarks, bbox, title_prefix=f"Shape: {shape}")
        else:
            print("Face not detected by FaceMesh. Showing segmentation/original.")
            visualize_analysis_results(orig_img, head_mask_img if head_mask_img is not None else orig_img, full_facemesh_img, None, None, title_prefix="Segmentation (No Face Mesh)")
    else: print(f"Could not load/process image: {IMAGE_PATH}. Status: {orig_img}")