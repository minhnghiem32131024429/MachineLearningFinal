import random
import os
import traceback
import PIL.Image
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import torch, cv2, numpy as np, matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import time
import argparse
import sys
import math
import matplotlib
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
matplotlib.use('Agg')
try:
    import clip
    import torch
    CLIP_AVAILABLE = True
    from ultralytics import YOLO
    POSE_MODEL_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not available. To use precise ball detection, install with: pip install clip")
    POSE_MODEL_AVAILABLE = False
    print("YOLOv8-Pose không khả dụng. Cài đặt với: pip install ultralytics")



def classify_sports_ball_with_clip(image, box, device=None):
    """
    Sử dụng CLIP để phân loại chính xác loại bóng từ vùng đã phát hiện là 'sports ball'

    Args:
        image: Ảnh numpy array (RGB)
        box: [x1, y1, x2, y2] - Tọa độ bounding box của bóng
        device: Thiết bị tính toán (cuda/cpu)

    Returns:
        String: Loại bóng cụ thể ("soccer ball", "basketball", ...)
    """
    if not CLIP_AVAILABLE:
        return "sports ball"

    # Xác định thiết bị nếu chưa được chỉ định
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tải model CLIP (chỉ tải một lần)
    if not hasattr(classify_sports_ball_with_clip, 'model'):
        print("Đang tải model CLIP...")
        classify_sports_ball_with_clip.model, classify_sports_ball_with_clip.preprocess = clip.load("ViT-B/32",
                                                                                                    device=device)
        print("Đã tải model CLIP thành công")

    model = classify_sports_ball_with_clip.model
    preprocess = classify_sports_ball_with_clip.preprocess

    # Cắt vùng ảnh chứa bóng từ box
    x1, y1, x2, y2 = [int(coord) for coord in box]
    # Thêm padding nhỏ xung quanh để đảm bảo lấy toàn bộ bóng
    padding = int(min(x2 - x1, y2 - y1) * 0.1)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)

    ball_img = image[y1:y2, x1:x2]

    # Chuyển thành định dạng PIL Image và tiền xử lý cho CLIP
    try:
        pil_img = PIL.Image.fromarray(ball_img.astype('uint8'))
        processed_img = preprocess(pil_img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh bóng: {str(e)}")
        return "sports ball"

    # Danh sách các mô tả về loại bóng (kết hợp nhiều biến thể để tăng độ chính xác)
    ball_descriptions = [
        # Bóng đá
        "a soccer ball", "a white and black soccer ball", "a football used in soccer games",
        "a FIFA soccer ball", "a round soccer ball with pentagonal patterns",

        # Bóng rổ
        "a basketball", "an orange basketball with black lines", "a ball used in basketball",
        "a Spalding basketball", "an NBA basketball",

        # Bóng tennis
        "a tennis ball", "a yellow-green tennis ball", "a small fuzzy ball used in tennis",
        "a Wilson tennis ball", "a bright yellow tennis ball",

        # Bóng chuyền
        "a volleyball", "a white volleyball with panels", "a ball used in volleyball games",
        "a Mikasa volleyball", "a white and blue volleyball",

        # Bóng chày
        "a baseball", "a white baseball with red stitching", "a small hard ball used in baseball",
        "a Major League baseball", "a leather baseball",

        # Bóng golf
        "a golf ball", "a small white golf ball with dimples", "a ball used in golf",
        "a Titleist golf ball", "a dimpled white golf ball",

        # Bóng rugby
        "a rugby ball", "an oval-shaped rugby ball", "a ball used in rugby",
        "an American football", "an NFL football",

        # Các loại bóng khác
        "a ping pong ball", "a small white table tennis ball", "a ball used in table tennis",
        "a bowling ball", "a heavy ball used for bowling", "a black bowling ball",
        "a beach ball", "a large inflatable ball", "a colorful beach ball"
    ]


    # Mã hóa các mô tả văn bản
    text_inputs = clip.tokenize(ball_descriptions).to(device)

    with torch.no_grad():
        # Mã hóa hình ảnh
        image_features = model.encode_image(processed_img)
        # Mã hóa văn bản
        text_features = model.encode_text(text_inputs)

        # Tính toán độ tương đồng giữa hình ảnh và văn bản
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Tìm mô tả có độ tương đồng cao nhất
        values, indices = similarity[0].topk(3)

    # Chọn mô tả có độ tương đồng cao nhất
    best_match_idx = indices[0].item()
    best_match = ball_descriptions[best_match_idx]

    # MỞ RỘNG MAPPING CÁC LOẠI BÓNG
    if "soccer" in best_match:
        return "soccer ball"
    elif "basketball" in best_match or "NBA" in best_match:
        return "basketball"
    elif "tennis" in best_match:
        return "tennis ball"
    elif "volleyball" in best_match:
        return "volleyball"
    elif "baseball" in best_match and "Major League" not in best_match:
        return "baseball"
    elif "golf" in best_match:
        return "golf ball"
    elif "rugby" in best_match or "American football" in best_match or "NFL" in best_match:
        return "american football"
    elif "ping pong" in best_match or "table tennis" in best_match:
        return "ping pong ball"
    elif "bowling" in best_match:
        return "bowling ball"
    elif "beach" in best_match:
        return "beach ball"

    # Nếu không chắc chắn, giữ nguyên nhãn gốc
    return "sports ball"

# DNN Face Detection Functions
def detect_faces_improved(image):
    """
    Phát hiện khuôn mặt với DNN model - cải thiện cho nhiều loại da và góc quay

    Args:
        image: Ảnh RGB (không phải BGR)

    Returns:
        faces: Danh sách các khuôn mặt dưới dạng (x, y, w, h)
    """
    print("Phát hiện khuôn mặt với DNN model...")

    # Đảm bảo ảnh đúng định dạng BGR (OpenCV)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Kiểm tra nếu đầu vào là RGB, chuyển sang BGR
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image.copy()

    # Lấy kích thước ảnh
    height, width = img_bgr.shape[:2]

    # Tạo thư mục cho model
    model_dir = "face_models"
    os.makedirs(model_dir, exist_ok=True)

    # Đường dẫn tới các file model
    model_files = {
        "prototxt": os.path.join(model_dir, "deploy.prototxt"),
        "model": os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    }

    # Kiểm tra và tải model nếu chưa có
    if not os.path.exists(model_files["prototxt"]) or not os.path.exists(model_files["model"]):
        print("Đang tải model phát hiện khuôn mặt...")

        # URLs cho model files
        urls = {
            "prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "model": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        }

        # Tải các file
        try:
            import urllib.request
            for name, url in urls.items():
                if not os.path.exists(model_files[name]):
                    print(f"Đang tải {name}...")
                    urllib.request.urlretrieve(url, model_files[name])
        except Exception as e:
            print(f"Lỗi khi tải model: {str(e)}")
            # Sử dụng haar cascade nếu không tải được DNN model
            return []

    # Tải model
    try:
        face_net = cv2.dnn.readNetFromCaffe(model_files["prototxt"], model_files["model"])

        # Chuẩn bị blob từ ảnh - quan trọng với preprocess
        # mean subtraction giúp cải thiện với các tông da khác nhau
        blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (300, 300),
                                     [104, 117, 123], False, False)

        # Đưa blob vào network
        face_net.setInput(blob)

        # Thực hiện phát hiện
        detections = face_net.forward()

        # Danh sách khuôn mặt phát hiện được
        faces = []

        # Ngưỡng tin cậy - có thể giảm xuống để phát hiện thêm khuôn mặt khó
        confidence_threshold = 0.6

        if not faces and confidence_threshold > 0.5:
            print("Không phát hiện khuôn mặt, giảm ngưỡng tin cậy...")
            confidence_threshold = 0.5

        # Duyệt qua các khuôn mặt phát hiện
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Lọc theo ngưỡng tin cậy
            if confidence > confidence_threshold:
                # Lấy tọa độ khuôn mặt
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x1, y1, x2, y2) = box.astype("int")

                # Đảm bảo tọa độ nằm trong ảnh
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                # Tính toán width và height
                w = x2 - x1
                h = y2 - y1

                # Thêm vào danh sách nếu kích thước hợp lý
                if w > 20 and h > 20:
                    faces.append((x1, y1, w, h))
                    print(f"Phát hiện khuôn mặt: {x1},{y1} - {w}x{h} (tin cậy: {confidence:.2f})")

        # Nếu không phát hiện được khuôn mặt nào, giảm ngưỡng và thử lại
        if not faces and confidence_threshold > 0.3:
            print("Không phát hiện khuôn mặt, giảm ngưỡng tin cậy...")
            confidence_threshold = 0.3

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (x1, y1, x2, y2) = box.astype("int")

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    w = x2 - x1
                    h = y2 - y1

                    if w > 20 and h > 20:
                        faces.append((x1, y1, w, h))
                        print(f"Phát hiện khuôn mặt (ngưỡng thấp): {x1},{y1} - {w}x{h} (tin cậy: {confidence:.2f})")

        return faces

    except Exception as e:
        print(f"Lỗi khi sử dụng DNN face detector: {str(e)}")
        print(f"Chi tiết: {traceback.format_exc()}")
        # Sử dụng haar cascade nếu DNN gặp lỗi
        return []




def select_best_face(faces, image):
    """
    Chọn khuôn mặt tốt nhất từ danh sách các khuôn mặt phát hiện được
    """
    if not faces:
        return None

    # Nếu chỉ có 1 khuôn mặt, trả về luôn
    if len(faces) == 1:
        return faces[0]

    # Tiêu chí chọn khuôn mặt:
    # 1. Khuôn mặt ở giữa ảnh
    # 2. Khuôn mặt có kích thước lớn
    # 3. Khuôn mặt có độ tương phản tốt

    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2

    best_face = None
    best_score = -1

    for (x, y, w, h) in faces:
        # Tính điểm tâm khuôn mặt
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Khoảng cách từ tâm khuôn mặt đến tâm ảnh (chuẩn hóa)
        distance_to_center = math.sqrt(((face_center_x - center_x) / width) ** 2 +
                                       ((face_center_y - center_y) / height) ** 2)

        # Kích thước tương đối của khuôn mặt
        relative_size = (w * h) / (width * height)

        # Tính độ tương phản của khuôn mặt
        face_roi = image[y:y + h, x:x + w]
        if len(face_roi.shape) == 3:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        contrast = np.std(face_roi)

        # Tính điểm số tổng hợp (các hệ số có thể điều chỉnh)
        # - Khuôn mặt gần tâm có điểm cao (1 - distance_to_center)
        # - Khuôn mặt lớn có điểm cao (relative_size)
        # - Khuôn mặt có độ tương phản cao có điểm cao (contrast/128)
        score = (1 - distance_to_center) * 0.5 + relative_size * 0.3 + (contrast / 128) * 0.2

        if score > best_score:
            best_score = score
            best_face = (x, y, w, h)

    return best_face

def get_joint_angle(p1, p2, p3):
    """Calculate angle between three points (joint angle)"""
    import numpy as np
    
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    return angle

def get_vertical_distance(p1, p2, p3, p4):
    """Get normalized vertical distance between two point pairs"""
    y1 = (p1[1] + p2[1])/2  # Average y of first pair
    y2 = (p3[1] + p4[1])/2  # Average y of second pair
    return abs(y1 - y2)

def get_horizontal_distance(p1, p2):
    """Get normalized horizontal distance between two points""" 
    return abs(p1[0] - p2[0])


def detect_baseball_patterns(keypoints, detected_equipment, left_arm_angle, right_arm_angle, spine_angle):
    """
    Detect baseball-specific patterns from pose and equipment
    
    Args:
        keypoints: List of body keypoints
        detected_equipment: List of detected sports equipment
        left_arm_angle: Angle of left arm
        right_arm_angle: Angle of right arm
        spine_angle: Angle of spine relative to vertical
    
    Returns:
        bool: True if baseball pattern detected
    """
    # Check for baseball-specific equipment
    if any(eq.lower() in ['baseball bat', 'baseball', 'baseball glove'] for eq in detected_equipment):
        # Batting stance detection
        if (45 <= left_arm_angle <= 135 and 45 <= right_arm_angle <= 135 and 
            abs(spine_angle) <= 30):
            return True
            
        # Pitching motion detection
        if (right_arm_angle >= 160 or left_arm_angle >= 160) and abs(spine_angle) <= 20:
            return True
            
    return False

def detect_swimming_patterns(keypoints, spine_angle, arm_spread):
    """
    Detect swimming-specific patterns from pose
    
    Args:
        keypoints: List of body keypoints
        spine_angle: Angle of spine relative to horizontal
        arm_spread: Angle between arms
    
    Returns:
        bool: True if swimming pattern detected
    """
    # Check for horizontal body position
    if abs(spine_angle - 90) <= 30:
        # Check for freestyle/butterfly arm position
        if arm_spread >= 120 and arm_spread <= 180:
            return True
            
        # Check for breaststroke position
        if arm_spread >= 60 and arm_spread <= 120:
            return True
            
    return False

def detect_volleyball_patterns(keypoints, detected_equipment, left_arm_angle, right_arm_angle, spine_angle):
    """
    Detect volleyball-specific patterns from pose and equipment
    
    Args:
        keypoints: List of body keypoints
        detected_equipment: List of detected sports equipment
        left_arm_angle: Angle of left arm
        right_arm_angle: Angle of right arm
        spine_angle: Angle of spine relative to vertical
    
    Returns:
        bool: True if volleyball pattern detected
    """
    # Check for volleyball equipment
    if any(eq.lower() == 'volleyball' for eq in detected_equipment):
        # Spiking motion detection
        if (right_arm_angle >= 150 or left_arm_angle >= 150) and abs(spine_angle) <= 30:
            return True
            
        # Setting motion detection
        if (90 <= right_arm_angle <= 180 and 90 <= left_arm_angle <= 180):
            return True
            
    return False

def detect_golf_patterns(keypoints, detected_equipment, spine_angle, arm_spread):
    """
    Detect golf-specific patterns from pose and equipment
    
    Args:
        keypoints: List of body keypoints
        detected_equipment: List of detected sports equipment
        spine_angle: Angle of spine relative to vertical
        arm_spread: Angle between arms
    
    Returns:
        bool: True if golf pattern detected
    """
    # Check for golf equipment
    if any(eq.lower() in ['golf club', 'golf ball'] for eq in detected_equipment):
        # Golf swing stance detection
        if (20 <= abs(spine_angle) <= 45 and 
            120 <= arm_spread <= 180):
            return True
            
        # Putting stance detection
        if abs(spine_angle) <= 30 and arm_spread <= 90:
            return True
            
    return False

def detect_rugby_patterns(keypoints, detected_equipment, shoulders_hip_dist, arm_spread):
    """
    Detect rugby-specific patterns from pose and equipment
    
    Args:
        keypoints: List of body keypoints
        detected_equipment: List of detected sports equipment
        shoulders_hip_dist: Distance between shoulders and hips
        arm_spread: Angle between arms
    
    Returns:
        bool: True if rugby pattern detected
    """
    # Check for rugby equipment
    if any(eq.lower() == 'rugby ball' for eq in detected_equipment):
        # Running with ball detection
        if shoulders_hip_dist >= 0.8 and arm_spread <= 90:
            return True
            
        # Passing motion detection
        if shoulders_hip_dist >= 0.6 and 90 <= arm_spread <= 180:
            return True
            
    return False

def detect_racket_sport_patterns(keypoints, detected_equipment, left_arm_angle, right_arm_angle, spine_angle):
    """
    Detect tennis and badminton specific patterns from pose and equipment
    
    Args:
        keypoints: List of body keypoints
        detected_equipment: List of detected sports equipment or empty list
        left_arm_angle: Angle of left arm relative to vertical
        right_arm_angle: Angle of right arm relative to vertical
        spine_angle: Angle of spine relative to vertical
    
    Returns:
        bool: True if tennis/badminton pattern detected
    """
    # First check if any racket equipment is detected
    if not any(eq.lower() in ['tennis racket', 'badminton racket', 'racket'] 
              for eq in (detected_equipment or [])):
        return False

    # Serving motion detection
    serving_detected = (
        (right_arm_angle >= 150 or left_arm_angle >= 150) and  # Raised arm for serve
        abs(spine_angle) <= 25  # Upright posture during serve
    )

    # Forehand swing detection
    forehand_detected = (
        (45 <= right_arm_angle <= 135 or 45 <= left_arm_angle <= 135) and  # Side arm position
        abs(spine_angle) <= 40  # Slight forward lean
    )

    # Backhand swing detection
    backhand_detected = (
        (80 <= right_arm_angle <= 160 or 80 <= left_arm_angle <= 160) and  # Cross-body position
        10 <= abs(spine_angle) <= 45  # Rotated stance
    )

    # Overhead smash detection
    smash_detected = (
        (right_arm_angle >= 160 or left_arm_angle >= 160) and  # High arm position
        abs(spine_angle) <= 30  # Slightly arched back
    )

    # Return true if any tennis/badminton pattern is detected
    return any([serving_detected, forehand_detected, backhand_detected, smash_detected])


def analyze_pose_patterns(keypoints, skeleton, sport_type=None, detected_equipment=None, img_data=None):
    """
    Comprehensive sport detection system with specific patterns for each supported sport
    """
    results = {
        'identified_sport': None,
        'confidence': 0.0,
        'pose_pattern': None,
        'evidence': []
    }

    if len(keypoints) < 10:
        return results

    try:
        # Get joint angles and positions
        left_arm_angle = get_joint_angle(keypoints[5], keypoints[7], keypoints[9])
        right_arm_angle = get_joint_angle(keypoints[6], keypoints[8], keypoints[10])
        left_leg_angle = get_joint_angle(keypoints[11], keypoints[13], keypoints[15])
        right_leg_angle = get_joint_angle(keypoints[12], keypoints[14], keypoints[16])
        
        # Additional measurements
        shoulders_hip_dist = get_vertical_distance(keypoints[5], keypoints[6], keypoints[11], keypoints[12])
        arm_spread = get_horizontal_distance(keypoints[9], keypoints[10])
        leg_spread = get_horizontal_distance(keypoints[15], keypoints[16])
        spine_angle = get_body_lean_angle(keypoints[5], keypoints[6], keypoints[11], keypoints[12])

        patterns = []

        # 1. SOCCER/FOOTBALL DETECTION
        if detect_soccer_patterns(keypoints, detected_equipment, left_leg_angle, right_leg_angle, 
                                arm_spread, leg_spread):
            patterns.extend([('Soccer', 0.95, 'soccer_kicking')])

        # 2. BASKETBALL DETECTION
        if detect_basketball_patterns(keypoints, detected_equipment, left_arm_angle, right_arm_angle,
                                   shoulders_hip_dist):
            patterns.extend([('Basketball', 0.90, 'basketball_shooting')])

        # 3. TENNIS/BADMINTON DETECTION
        if detect_racket_sport_patterns(keypoints, detected_equipment, left_arm_angle, right_arm_angle,
                                      spine_angle):
            if 'tennis racket' in (detected_equipment or []):
                patterns.extend([('Tennis', 0.92, 'tennis_swing')])
            else:
                patterns.extend([('Badminton', 0.85, 'badminton_swing')])

        # 4. CYCLING DETECTION
        if detect_cycling_patterns(keypoints, detected_equipment, spine_angle, shoulders_hip_dist):
            patterns.extend([('Cycling', 0.95, 'cycling_position')])

        # 5. RUNNING/TRACK DETECTION
        if detect_running_patterns(keypoints, detected_equipment, left_leg_angle, right_leg_angle,
                                 arm_spread, spine_angle):
            patterns.extend([('Running', 0.85, 'running_stride')])

        # 6. BASEBALL DETECTION
        if detect_baseball_patterns(keypoints, detected_equipment, left_arm_angle, right_arm_angle,
                                  spine_angle):
            patterns.extend([('Baseball', 0.88, 'baseball_swing')])

        # 7. SWIMMING DETECTION
        if detect_swimming_patterns(keypoints, spine_angle, arm_spread):
            patterns.extend([('Swimming', 0.87, 'swimming_stroke')])

        # 8. VOLLEYBALL DETECTION
        if detect_volleyball_patterns(keypoints, detected_equipment, left_arm_angle, right_arm_angle,
                                    spine_angle):
            patterns.extend([('Volleyball', 0.86, 'volleyball_spike')])

        # 9. GOLF DETECTION
        if detect_golf_patterns(keypoints, detected_equipment, spine_angle, arm_spread):
            patterns.extend([('Golf', 0.93, 'golf_swing')])

        # 10. RUGBY DETECTION
        if detect_rugby_patterns(keypoints, detected_equipment, shoulders_hip_dist, arm_spread):
            patterns.extend([('Rugby', 0.89, 'rugby_running')])

        # Select best pattern based on confidence and evidence
        if patterns:
            patterns.sort(key=lambda x: x[1], reverse=True)
            best_pattern = patterns[0]
            results.update({
                'identified_sport': best_pattern[0],
                'confidence': best_pattern[1],
                'pose_pattern': best_pattern[2],
                'all_patterns': patterns
            })

    except Exception as e:
        print(f"Error analyzing pose patterns: {str(e)}")
        results['evidence'].append(f'Error: {str(e)}')

    return results

# Sport-specific detection functions
def detect_soccer_patterns(keypoints, equipment, left_leg_angle, right_leg_angle, arm_spread, leg_spread):
    """Soccer-specific pattern detection"""
    indicators = 0
    
    # Kicking motion
    if max(left_leg_angle, right_leg_angle) > 120:
        indicators += 2
    
    # Balance position
    if 45 < arm_spread < 120:
        indicators += 1
        
    # Wide stance
    if leg_spread > 0.3:
        indicators += 1
        
    # Ball presence
    if equipment and any(x in ['soccer ball', 'football'] for x in equipment):
        indicators += 2
        
    return indicators >= 3

def detect_basketball_patterns(keypoints, equipment, left_arm_angle, right_arm_angle, shoulders_hip_dist):
    """Basketball-specific pattern detection"""
    indicators = 0
    
    # Shooting motion
    if max(left_arm_angle, right_arm_angle) > 150:
        indicators += 2
        
    # Jump shot position
    if shoulders_hip_dist > 0.25:
        indicators += 1
        
    # Ball presence
    if equipment and 'basketball' in equipment:
        indicators += 2
        
    return indicators >= 3

def detect_cycling_patterns(keypoints, equipment, spine_angle, shoulders_hip_dist):
    """Cycling-specific pattern detection"""
    indicators = 0
    
    # Forward lean
    if spine_angle < 45:
        indicators += 2
        
    # Compact position
    if shoulders_hip_dist < 0.15:
        indicators += 1
        
    # Bicycle presence
    if equipment and 'bicycle' in equipment:
        indicators += 3
        
    return indicators >= 4

# Add similar functions for other sports...
def detect_running_patterns(keypoints, equipment, left_leg_angle, right_leg_angle, arm_spread, spine_angle):
    """Running-specific pattern detection"""
    indicators = 0
    
    # Alternating leg motion
    if abs(left_leg_angle - right_leg_angle) > 45:
        indicators += 2
        
    # Upright posture
    if spine_angle > 75:
        indicators += 1
        
    # Arm pump motion
    if arm_spread > 0.3:
        indicators += 1
        
    # No equipment dependence for running
    if not equipment:
        indicators += 1
        
    return indicators >= 3

# ... Add other sport detection functions ...

def get_body_lean_angle(left_shoulder, right_shoulder, left_hip, right_hip):
    """Calculate body lean angle relative to vertical"""
    shoulder_mid = ((left_shoulder[0] + right_shoulder[0])/2, 
                   (left_shoulder[1] + right_shoulder[1])/2)
    hip_mid = ((left_hip[0] + right_hip[0])/2, 
               (left_hip[1] + right_hip[1])/2)
    
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    angle = abs(np.degrees(np.arctan2(dx, dy)))
    
    return angle


def get_body_lean_angle(left_shoulder, right_shoulder, left_hip, right_hip):
    """Calculate the body's forward/backward lean angle"""
    # Calculate midpoints
    shoulder_mid = ((left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2)
    hip_mid = ((left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2)
    
    # Calculate angle with vertical
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    angle = abs(np.degrees(np.arctan2(dx, dy)))
    
    return angle

def get_joint_angle(p1, p2, p3):
    """Calculate angle between three points (joint angle)"""
    import numpy as np
    
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    return angle

def get_vertical_distance(p1, p2, p3, p4):
    """Get normalized vertical distance between two point pairs"""
    y1 = (p1[1] + p2[1])/2  # Average y of first pair
    y2 = (p3[1] + p4[1])/2  # Average y of second pair
    return abs(y1 - y2)

def get_horizontal_distance(p1, p2):
    """Get normalized horizontal distance between two points""" 
    return abs(p1[0] - p2[0])

def check_dependencies():
    required_packages = {
        'ultralytics': 'ultralytics',
        'mtcnn': 'mtcnn',
        'timm': 'timm',  # Thêm timm cho MiDaS
        'torch': 'torch',
        'torchvision': 'torchvision',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'ultralytics': 'ultralytics',
        'scipy': 'scipy',
        'PIL': 'pillow',
        'clip': 'git+https://github.com/openai/CLIP.git',
        'ftfy': 'ftfy',
        'regex': 'regex'
    }

    missing_packages = []

    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {module} already installed")
        except ImportError:
            print(f"✗ {module} missing - will install {package}")
            missing_packages.append(package)

    if missing_packages:
        print("\nInstalling missing packages...")
        for package in missing_packages:
            print(f"Installing {package}...")
            os.system(f"pip install {package}")
            print(f"Installed {package}")

        # Nếu có gói nào đã được cài đặt, reload môi trường
        if missing_packages:
            print("\nReloading environment with new packages...")
            # Force reload modules in case they were partially loaded
            for module in sys.modules.copy():
                if any(module.startswith(pkg.split('.')[0]) for pkg in required_packages.keys()):
                    try:
                        del sys.modules[module]
                    except:
                        pass

    print("All dependencies checked and installed.")


# Setup device and load models
def load_models():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load MiDaS depth model with explicit trust_repo
        print("Loading MiDaS depth model...")
        midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
        midas.to(device)
        midas.eval()
        print("MiDaS model loaded successfully.")

        # Load YOLO model for object detection
        print("Loading YOLO model...")
        from ultralytics import YOLO
        yolo = YOLO("yolov8x.pt")  # Load the largest YOLOv8 model
        print("YOLO model loaded successfully.")
        yolo_seg = YOLO("yolov8x-seg.pt")
        print("YOLOv8-seg model loaded successfully.")

        return midas, yolo, yolo_seg, device

    except Exception as e:
        print(f"Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    # Resize keeping aspect ratio
    width, height = img.size
    ratio = min(520 / width, 520 / height)
    new_size = (int(width * ratio), int(height * ratio))
    resized_img = img.resize(new_size, Image.LANCZOS)

    # Get the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Process for MiDaS
    midas_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((384, 384), antialias=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    midas_input = midas_transform(img).unsqueeze(0).to(device)

    return {
        'original': img, 'array': img_array,
        'resized': resized_img, 'resized_array': np.array(resized_img),
        'midas_input': midas_input
    }


def detect_sports_objects(yolo, img_data):
    """Detect sports-related objects using YOLO"""
    results = yolo(img_data['resized_array'], conf=0.25)

    # Extract detections - CHỈ CÁC LỚP YOLO THỰC SỰ CÓ
    sports_classes = [
        'person',           # Người
        'sports ball',      # Bóng thể thao (sẽ dùng CLIP phân loại chi tiết)
        'tennis racket',    # Vợt tennis
        'baseball bat',     # Gậy baseball
        'baseball glove',   # Găng tay baseball
        'frisbee',         # Đĩa bay
        'skis',            # Ván trượt tuyết
        'snowboard',       # Ván trượt tuyết đơn
        'surfboard',       # Ván lướt sóng
        'bicycle',         # Xe đạp
        'motorcycle',      # Xe máy
        'kite',            # Diều
        'skateboard',      # Ván trượt
        'bottle',          # Chai nước (phụ kiện thể thao)
        'backpack',        # Ba lô thể thao
        'handbag',         # Túi thể thao
        'umbrella',        # Ô (cho golf)
        'tie',             # Cà vạt (trang phục thể thao chính thức)
        'suitcase',        # Vali đựng đồ thể thao
        'cup'              # Cốc/ly (giải thưởng hoặc nước uống)
    ]

    result = results[0]  # First image result

    detections = {
        'boxes': [],
        'classes': [],
        'scores': [],
        'sports_objects': 0,
        'athletes': 0
    }

    if hasattr(result, 'boxes') and len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = result.names[cls]

            detections['boxes'].append([x1, y1, x2, y2])
            detections['classes'].append(class_name)
            detections['scores'].append(conf)

            # Sử dụng CLIP để phân loại chính xác loại bóng
            if class_name == 'sports ball' and CLIP_AVAILABLE and conf > 0.4:
                try:
                    # Xác định loại bóng cụ thể bằng CLIP
                    specific_ball = classify_sports_ball_with_clip(img_data['resized_array'], [x1, y1, x2, y2])
                    print(f"CLIP phân loại: 'sports ball' -> '{specific_ball}'")

                    # Cập nhật lại class_name với loại bóng xác định được
                    class_name = specific_ball
                    detections['classes'][-1] = class_name
                except Exception as e:
                    print(f"Lỗi khi phân loại bóng với CLIP: {str(e)}")

            if class_name == 'person':
                detections['athletes'] += 1
            if class_name in sports_classes:
                detections['sports_objects'] += 1

    return detections

def generate_depth_map(midas, img_data):
    with torch.no_grad():
        depth_map = midas(img_data['midas_input'])
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=img_data['array'].shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

    normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-5)

    depth_8bit = (normalized_depth * 255).astype(np.uint8)
    _, depth_mask = cv2.threshold(depth_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea) if contours else None

    return normalized_depth, depth_mask, main_contour


# Hàm phân tích độ sắc nét của đối tượng (MỚI)
def analyze_object_sharpness(image, boxes):
    """
    Phân tích độ sắc nét của các đối tượng được phát hiện
    Cải thiện: Sử dụng nhiều phương pháp đánh giá sharpness
    """
    import cv2
    import numpy as np

    if len(boxes) == 0:
        return []

    # Convert to grayscale nếu cần
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    sharpness_scores = []
    sharpness_details = []
    h, w = gray.shape

    for box in boxes:
        try:
            # Đảm bảo box nằm trong ảnh
            x1, y1, x2, y2 = map(int, box[:4])
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            # Trích xuất vùng quan tâm
            roi = gray[y1:y2, x1:x2]

            # Bỏ qua vùng quá nhỏ
            if roi.shape[0] < 10 or roi.shape[1] < 10:
                sharpness_scores.append(0.0)
                continue

            # **PHƯƠNG PHÁP 1: Laplacian Variance (có lọc noise)**
            # Áp dụng Gaussian blur nhẹ để giảm noise
            roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
            laplacian = cv2.Laplacian(roi_blur, cv2.CV_64F)
            laplacian_var = laplacian.var()

            # **PHƯƠNG PHÁP 2: Sobel Gradient Magnitude**
            sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
            sobel_mean = np.mean(sobel_magnitude)

            # **PHƯƠNG PHÁP 3: Brenner Gradient**
            # Tính gradient theo phương ngang
            if roi.shape[1] > 2:
                brenner = np.sum((roi[:, 2:] - roi[:, :-2]) ** 2)
                brenner_norm = brenner / (roi.shape[0] * (roi.shape[1] - 2))
            else:
                brenner_norm = 0

            # **PHƯƠNG PHÁP 4: Tenengrad (Sobel based)**
            sobelx_thresh = np.where(np.abs(sobelx) > 10, sobelx, 0)
            sobely_thresh = np.where(np.abs(sobely) > 10, sobely, 0)
            tenengrad = np.sum(sobelx_thresh ** 2 + sobely_thresh ** 2)
            tenengrad_norm = tenengrad / (roi.shape[0] * roi.shape[1])

            # **TỔNG HỢP ĐIỂM SHARPNESS**
            # Normalize từng thành phần
            laplacian_norm = min(laplacian_var / 1000.0, 1.0)  # Cap ở 1.0
            sobel_norm = min(sobel_mean / 50.0, 1.0)
            brenner_norm = min(brenner_norm / 100.0, 1.0)
            tenengrad_norm = min(tenengrad_norm / 500.0, 1.0)

            # Weighted combination
            final_score = (
                    0.3 * laplacian_norm +  # Laplacian variance
                    0.3 * sobel_norm +  # Sobel gradient
                    0.2 * brenner_norm +  # Brenner gradient
                    0.2 * tenengrad_norm  # Tenengrad
            )

            # **ĐIỀU CHỈNH THEO KÍCH THƯỚC**
            # Vùng lớn hơn thường có điểm sharpness ổn định hơn
            area = (x2 - x1) * (y2 - y1)
            size_factor = min(np.sqrt(area) / 100.0, 1.2)  # Bonus cho vùng lớn

            final_score *= size_factor
            final_score = min(final_score, 1.0)  # Cap ở 1.0

            sharpness_scores.append(float(final_score))

            # THÊM ĐOẠN NÀY - Lưu chi tiết cho debugging
            sharpness_details.append({
                'laplacian_var': laplacian_var,
                'sobel_mean': sobel_mean,
                'brenner_norm': brenner_norm,
                'tenengrad_norm': tenengrad_norm,
                'combined_score': final_score
            })

        except Exception as e:
            print(f"Error calculating sharpness for box {box}: {e}")
            sharpness_scores.append(0.0)
            sharpness_details.append({
                'laplacian_var': 0,
                'sobel_mean': 0,
                'brenner_norm': 0,
                'tenengrad_norm': 0,
                'combined_score': 0
            })

    return sharpness_scores, sharpness_details

def create_sharpness_heatmap(image, sharpness_map=None):
    """
    Tạo heatmap độ sắc nét cho toàn bộ ảnh
    """
    try:
        import cv2
        import numpy as np

        # Đảm bảo image không None và có shape hợp lệ
        if image is None or len(image.shape) < 2:
            print("Invalid image input for sharpness heatmap")
            return image.copy() if image is not None else np.zeros((100, 100, 3), dtype=np.uint8), None

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Tính Laplacian cho toàn ảnh với kernel lớn hơn
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        laplacian_abs = np.abs(laplacian)

        # Làm mịn để tạo heatmap
        heatmap = cv2.GaussianBlur(laplacian_abs, (21, 21), 0)

        # Normalize về 0-255
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_norm = heatmap_norm.astype(np.uint8)

        # Áp dụng colormap JET
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        # Chuyển về RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay lên ảnh gốc
        if len(image.shape) == 3:
            overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
        else:
            image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            overlay = cv2.addWeighted(image_colored, 0.6, heatmap_colored, 0.4, 0)

        print("Sharpness heatmap created successfully")
        return overlay, heatmap_norm

    except Exception as e:
        print(f"Error in create_sharpness_heatmap: {e}")
        # Trả về ảnh gốc và None nếu có lỗi
        if image is not None:
            return image.copy(), None
        else:
            return np.zeros((100, 100, 3), dtype=np.uint8), None

def analyze_sports_scene(detections, depth_map, img_data, yolo_seg=None):
    """Analyze the sports scene based on detected objects and depth"""
    height, width = depth_map.shape[:2]

    # Phần phân tích phân bố người chơi (giữ nguyên)
    player_positions = []
    for i, cls in enumerate(detections['classes']):
        if cls == 'person':
            x1, y1, x2, y2 = detections['boxes'][i]
            # Scale box coordinates to match depth map dimensions
            x1 = int(x1 * depth_map.shape[1] / img_data['resized_array'].shape[1])
            y1 = int(y1 * depth_map.shape[0] / img_data['resized_array'].shape[0])
            x2 = int(x2 * depth_map.shape[1] / img_data['resized_array'].shape[1])
            y2 = int(y2 * depth_map.shape[0] / img_data['resized_array'].shape[0])

            center_x = (x1 + x2) / 2 / width
            center_y = (y1 + y2) / 2 / height
            player_positions.append((center_x, center_y))

    # Tính độ phân tán người chơi (giữ nguyên)
    player_dispersion = 0
    if len(player_positions) > 1:
        total_distance = 0
        count = 0
        for i in range(len(player_positions)):
            for j in range(i + 1, len(player_positions)):
                p1 = player_positions[i]
                p2 = player_positions[j]
                dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                total_distance += dist
                count += 1
        if count > 0:
            player_dispersion = total_distance / count

    # Phân tích độ sắc nét của các đối tượng (giữ nguyên)
    image = img_data['resized_array']
    sharpness_scores, sharpness_details = analyze_object_sharpness(image, detections['boxes'])
    print(f"Sharpness scores: {[f'{score:.2f}' for score in sharpness_scores]}")

    # PHẦN CẢI TIẾN: Xác định đối tượng chính (key_subjects)
    key_subjects = []

    # Tính kích thước tối thiểu (3% diện tích ảnh)
    min_size_threshold = 0.03 * (img_data['resized_array'].shape[0] * img_data['resized_array'].shape[1])

    # Trích xuất thông tin môn thể thao (nếu có)
    sport_type = "unknown"
    if 'composition_analysis' in img_data and 'sport_type' in img_data['composition_analysis']:
        sport_type = img_data['composition_analysis']['sport_type'].lower()

    for i, box in enumerate(detections['boxes']):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        area_ratio = area / (img_data['resized_array'].shape[0] * img_data['resized_array'].shape[1])

        # Scale coordinates to match depth map dimensions
        depth_x1 = int(x1 * depth_map.shape[1] / img_data['resized_array'].shape[1])
        depth_y1 = int(y1 * depth_map.shape[0] / img_data['resized_array'].shape[0])
        depth_x2 = int(x2 * depth_map.shape[1] / img_data['resized_array'].shape[1])
        depth_y2 = int(y2 * depth_map.shape[0] / img_data['resized_array'].shape[0])

        # Ensure coordinates are within depth map bounds
        depth_x1 = max(0, min(depth_x1, depth_map.shape[1] - 1))
        depth_y1 = max(0, min(depth_y1, depth_map.shape[0] - 1))
        depth_x2 = max(0, min(depth_x2, depth_map.shape[1] - 1))
        depth_y2 = max(0, min(depth_y2, depth_map.shape[0] - 1))

        # Create mask with depth map dimensions
        mask = np.zeros((depth_map.shape[0], depth_map.shape[1]), dtype=np.uint8)
        if depth_y2 > depth_y1 and depth_x2 > depth_x1:  # Ensure box has valid dimensions
            mask[depth_y1:depth_y2, depth_x1:depth_x2] = 1

        # Calculate average depth
        obj_depth = np.mean(depth_map[mask > 0]) if np.sum(mask) > 0 else 0

        # Lấy độ sắc nét
        sharpness = sharpness_scores[i] if i < len(sharpness_scores) else 0

        # Tính vị trí trung tâm và khoảng cách đến trung tâm ảnh
        center_x = (x1 + x2) / 2 / img_data['resized_array'].shape[1]
        center_y = (y1 + y2) / 2 / img_data['resized_array'].shape[0]
        center_dist = np.sqrt((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2)

        # Tạo thông tin đối tượng
        subject_info = {
            'class': detections['classes'][i],
            'box': box,
            'area': area,
            'area_ratio': area_ratio,
            'depth': obj_depth,
            'sharpness': sharpness,
            'sharpness_details': sharpness_details[i] if i < len(sharpness_details) else None,
            'position': (center_x, center_y),
            'center_dist': center_dist
        }

        # CẢI TIẾN 1: Điều chỉnh tỷ trọng các yếu tố
        # Kích thước và vị trí (60% thay vì 40%)
        position_weight = 0.2  # Trọng số cho vị trí trung tâm
        size_weight = 0.4  # Trọng số cho kích thước

        # Giảm tỷ trọng độ sắc nét (20% thay vì 40%)
        sharpness_weight = 0.2

        # Giữ nguyên tỷ trọng độ sâu (20%)
        depth_weight = 0.2

        # Tính điểm cho vị trí (càng gần trung tâm càng cao)
        position_score = (1 - min(1.0, center_dist * 1.5)) * position_weight

        # Tính điểm cho kích thước
        size_score = area_ratio * size_weight

        # Tính điểm cho độ sắc nét
        sharpness_score = sharpness * sharpness_weight

        # Tính điểm cho độ sâu (đối tượng gần hơn có điểm cao hơn)
        depth_score = (1 - obj_depth) * depth_weight

        # CẢI TIẾN 2: Ưu tiên đối tượng người và kích thước lớn
        class_multiplier = 1.0  # Mặc định

        # Nếu là người, nhân hệ số lớn (3.5x)
        if detections['classes'][i] == 'person':
            class_multiplier *= 3.5

        # Nếu kích thước đối tượng đủ lớn (>5% diện tích ảnh), thêm điểm
        if area_ratio > 0.05:
            class_multiplier *= 1.5

        # Lọc kích thước tối thiểu (3% diện tích ảnh)
        if area < min_size_threshold:
            class_multiplier *= 0.5  # Giảm 50% điểm cho đối tượng quá nhỏ

        # CẢI TIẾN 3: Logic đặc thù cho môn thể thao
        sport_bonus = 1.0

        # Đối với trượt tuyết, ưu tiên đối tượng ở phía trước/dưới (y lớn hơn)
        if "ski" in sport_type or "snow" in sport_type:
            # Đối tượng càng thấp (y lớn) càng có ưu thế
            if center_y > 0.6:  # Nằm phía dưới ảnh
                sport_bonus *= 1.3

        # Tổng hợp điểm số với trọng số mới và các hệ số điều chỉnh
        subject_info['prominence'] = (
                                                 position_score + size_score + sharpness_score + depth_score) * class_multiplier * sport_bonus

        # Lưu thông tin tính toán để debug
        subject_info['debug'] = {
            'position_score': position_score,
            'size_score': size_score,
            'sharpness_score': sharpness_score,
            'depth_score': depth_score,
            'class_multiplier': class_multiplier,
            'sport_bonus': sport_bonus
        }

        key_subjects.append(subject_info)

    # Sắp xếp theo prominence
    key_subjects.sort(key=lambda x: x['prominence'], reverse=True)

    # In thông tin debug cho các đối tượng hàng đầu
    if key_subjects:
        print(f"\nĐối tượng chính: {key_subjects[0]['class']}, Prominence: {key_subjects[0]['prominence']:.3f}")
        print(
            f"Kích thước: {key_subjects[0]['area_ratio'] * 100:.1f}% ảnh, Độ sắc nét: {key_subjects[0]['sharpness']:.2f}")
        print(f"Vị trí: {key_subjects[0]['position']}, Khoảng cách đến trung tâm: {key_subjects[0]['center_dist']:.2f}")

        if len(key_subjects) > 1:
            print(f"\nĐối tượng thứ 2: {key_subjects[1]['class']}, Prominence: {key_subjects[1]['prominence']:.3f}")
            print(
                f"Kích thước: {key_subjects[1]['area_ratio'] * 100:.1f}% ảnh, Độ sắc nét: {key_subjects[1]['sharpness']:.2f}")

    # Tạo biến sports_analysis
    sports_analysis = {
        'player_count': detections['athletes'],
        'player_positions': player_positions,
        'player_dispersion': player_dispersion,
        'key_subjects': key_subjects[:5] if key_subjects else [],
        'sharpness_scores': sharpness_scores
    }

    # Phân đoạn main subject nếu tìm thấy và là người
    if key_subjects and key_subjects[0]['class'] == 'person':
        main_subject_box = key_subjects[0]['box']
        print("Thực hiện phân đoạn main subject...")
        main_subject_mask = segment_main_subject(img_data['resized_array'], yolo_seg, main_subject_box)
        if main_subject_mask is not None:
            print(f"Đã tìm thấy mask cho main subject, kích thước: {main_subject_mask.shape}")
        else:
            print("Không tìm được mask phù hợp cho main subject")

        # Lưu mask vào kết quả phân tích
        sports_analysis['main_subject_mask'] = main_subject_mask

    return sports_analysis

def analyze_action_quality(detections, img_data):
    """Analyze action quality in sports image with improved approach"""
    height, width = img_data['resized_array'].shape[:2]

    # 1. Check for sports equipment
    has_equipment = False
    equipment_types = []
    for cls in detections['classes']:
        if cls in ['sports ball', 'tennis racket', 'baseball bat', 'baseball glove',
                   'skateboard', 'surfboard', 'tennis ball', 'frisbee', 'skis', 'snowboard']:
            has_equipment = True
            if cls not in equipment_types:
                equipment_types.append(cls)

    # 2. Analyze individual posture instead of comparing multiple people
    action_posture_score = 0
    dynamic_posture_count = 0
    total_players = 0

    for i, cls in enumerate(detections['classes']):
        if cls == 'person':
            total_players += 1
            x1, y1, x2, y2 = detections['boxes'][i]

            # a. Calculate height/width ratio
            aspect_ratio = (y2 - y1) / (x2 - x1) if (x2 - x1) > 0 else 0

            # b. Evaluate posture based on aspect ratio
            # Non-typical posture (jumping, crouching, lying...)
            if aspect_ratio < 1.2 or aspect_ratio > 2.5:
                dynamic_posture_count += 1

            # c. Calculate relative area (larger = closer action)
            area_ratio = ((y2 - y1) * (x2 - x1)) / (height * width)
            if area_ratio > 0.2:  # Athlete takes up large area, often close action
                action_posture_score += 0.2

    # If there are people in non-typical postures, that could be dynamic action
    if total_players > 0:
        dynamic_posture_ratio = dynamic_posture_count / total_players
        action_posture_score += dynamic_posture_ratio * 0.5

    # 3. Calculate improved action_level
    action_level = 0

    # If there's sports equipment
    if has_equipment:
        action_level += 0.4

    # Add points from posture analysis
    action_level += min(0.6, action_posture_score)

    # 4. Classify action quality
    return {
        'has_equipment': has_equipment,
        'equipment_types': equipment_types,
        'dynamic_posture_score': action_posture_score,
        'dynamic_posture_count': dynamic_posture_count,
        'total_players': total_players,
        'action_level': action_level,
        'action_quality': "High" if action_level > 0.7 else
        "Medium" if action_level > 0.3 else "Low"
    }


def verify_face(face_img):
    """Phát hiện khuôn mặt không hợp lệ với các tiêu chí nghiêm ngặt hơn"""
    try:
        # 1. Kiểm tra kích thước tối thiểu
        if face_img.shape[0] < 20 or face_img.shape[1] < 20:
            return False, "Khuôn mặt quá nhỏ (nhỏ hơn 20px)"

        # 2. Kiểm tra tỷ lệ khung hình
        h, w = face_img.shape[:2]
        aspect_ratio = h / w
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False, "Tỷ lệ khuôn mặt bất thường"

        # 3. Kiểm tra độ tương phản - khuôn mặt thực phải có độ tương phản
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        std_dev = np.std(gray)
        if std_dev < 25:  # Tăng ngưỡng (trước là 10)
            return False, "Độ tương phản quá thấp, có thể không phải khuôn mặt"

        # 4. MỚI: Kiểm tra kết cấu khuôn mặt sử dụng bộ lọc Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        avg_edge_strength = np.mean(sobel_mag)

        # Khuôn mặt thực phải có cạnh rõ ràng (mắt, mũi, miệng)
        if avg_edge_strength < 10.0:
            return False, "Không phát hiện đặc điểm khuôn mặt rõ ràng"

        # 5. MỚI: Kiểm tra vùng mắt - khuôn mặt thật phải có vùng mắt có độ tương phản
        eye_region_h = int(h * 0.3)
        eye_region = gray[:eye_region_h, :]
        eye_std = np.std(eye_region)

        if eye_std < 20:
            return False, "Không phát hiện vùng mắt rõ ràng"

        # 6. MỚI: Kiểm tra hướng khuôn mặt bằng cách tính toán phân phối gradient
        # Nếu khuôn mặt quay lưng, gradient sẽ không đồng đều
        gradient_y_ratio = np.mean(np.abs(sobely)) / (np.mean(np.abs(sobelx)) + 1e-5)
        if gradient_y_ratio < 0.5:
            return False, "Có thể khuôn mặt đang quay đi"

        return True, "Khuôn mặt hợp lệ"

    except Exception as e:
        return False, f"Lỗi xác thực: {str(e)}"

def analyze_sports_environment(img_data, depth_map=None):
    """
    Phân tích môi trường/bối cảnh thể thao dựa trên đặc điểm màu sắc, kết cấu và cấu trúc

    Args:
        img_data: Dict chứa ảnh gốc và ảnh đã resize
        depth_map: Bản đồ độ sâu (nếu có)

    Returns:
        Dict: Thông tin về môi trường thể thao và xác suất từng môn
    """
    # Lấy ảnh đã resize để phân tích
    image = img_data['resized_array']
    height, width = image.shape[:2]

    # Kết quả chứa xác suất các môn thể thao
    env_results = {
        'detected_environments': [],
        'sport_probabilities': {},
        'dominant_colors': [],
        'surface_type': 'unknown',
        'confidence': 0.0
    }

    # Phân tích màu sắc đặc trưng
    # Chuyển sang không gian màu HSV để phân tích tốt hơn
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # 1. Phát hiện màu sắc đặc trưng
    # Tạo mask cho từng vùng màu quan trọng

    # Xanh nước (bơi lội)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    blue_ratio = np.sum(blue_mask > 0) / (height * width)

    # Xanh cỏ (sân cỏ - bóng đá, điền kinh)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / (height * width)

    # Đỏ/nâu (sân đất nện - tennis, điền kinh)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    red_mask = red_mask1 | red_mask2
    red_ratio = np.sum(red_mask > 0) / (height * width)

    # Màu đen/xám đậm (đường chạy nhựa)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 30, 80])
    dark_mask = cv2.inRange(hsv_img, lower_dark, upper_dark)
    dark_ratio = np.sum(dark_mask > 0) / (height * width)

    # Màu trắng (sân võ thuật, sàn boxing)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv_img, lower_white, upper_white)
    white_ratio = np.sum(white_mask > 0) / (height * width)

    # Lưu tỷ lệ màu chính
    color_ratios = {
        'blue': blue_ratio,
        'green': green_ratio,
        'red': red_ratio,
        'dark': dark_ratio,
        'white': white_ratio
    }

    # Lấy 2 màu chiếm tỷ lệ cao nhất
    dominant_colors = sorted(color_ratios.items(), key=lambda x: x[1], reverse=True)[:2]
    env_results['dominant_colors'] = dominant_colors

    # 2. Phân tích kết cấu sân đấu

    # Chuyển sang grayscale cho phân tích kết cấu
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Phát hiện cạnh với Canny
    edges = cv2.Canny(gray, 50, 150)

    # Phát hiện đường thẳng với Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Đếm số đường thẳng ngang và dọc
    horizontal_lines = 0
    vertical_lines = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Tính góc của đường
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Phân loại dựa trên góc
            if angle < 30 or angle > 150:
                horizontal_lines += 1
            elif angle > 60 and angle < 120:
                vertical_lines += 1

    # 3. Suy đoán môi trường thể thao dựa trên các đặc điểm
    sport_probs = {}

    # Bơi lội
    if blue_ratio > 0.3 and horizontal_lines >= 3:
        sport_probs['Swimming'] = min(1.0, blue_ratio * 1.5)
        env_results['detected_environments'].append('swimming pool')
        env_results['surface_type'] = 'water'

    # Sân cỏ (bóng đá, rugby)
    if green_ratio > 0.4:
        sport_probs['Soccer'] = min(1.0, green_ratio * 1.2)
        sport_probs['Rugby'] = min(1.0, green_ratio * 1.0)
        env_results['detected_environments'].append('grass field')
        env_results['surface_type'] = 'grass'

    # Sân tennis đất nện
    if red_ratio > 0.3 and horizontal_lines >= 2 and vertical_lines >= 2:
        sport_probs['Tennis'] = min(1.0, red_ratio * 1.3)
        env_results['detected_environments'].append('clay court')
        env_results['surface_type'] = 'clay'

    # Đường chạy điền kinh
    if dark_ratio > 0.2 and horizontal_lines >= 3 and vertical_lines <= 2:
        sport_probs['Track and Field'] = min(1.0, dark_ratio * 1.2)
        sport_probs['Running'] = min(1.0, dark_ratio * 1.3)
        env_results['detected_environments'].append('running track')
        env_results['surface_type'] = 'track'

    # Sàn đấu Boxing/UFC/võ thuật
    if white_ratio > 0.3 and dark_ratio < 0.3:
        if horizontal_lines <= 3 and vertical_lines <= 3:
            env_canvas_ratio = 0.0
            if depth_map is not None:
                # Dùng depth map để xác định phần sàn đấu được nâng cao
                # Đây là logic đơn giản, có thể cải tiến thêm
                center_depth = depth_map[height // 3:2 * height // 3, width // 3:2 * width // 3]
                border_depth = np.concatenate([
                    depth_map[:height // 3, :],  # Trên
                    depth_map[2 * height // 3:, :],  # Dưới
                    depth_map[height // 3:2 * height // 3, :width // 3],  # Trái
                    depth_map[height // 3:2 * height // 3, 2 * width // 3:]  # Phải
                ])

                if np.mean(center_depth) < np.mean(border_depth):
                    env_canvas_ratio = 0.3

            sport_probs['Boxing'] = min(1.0, white_ratio * 0.8 + env_canvas_ratio)
            sport_probs['Martial Arts'] = min(1.0, white_ratio * 0.7 + env_canvas_ratio)
            env_results['detected_environments'].append('fighting ring/mat')
            env_results['surface_type'] = 'canvas'

    # Sân bóng rổ
    if (dark_ratio > 0.2 or red_ratio > 0.2) and horizontal_lines >= 2 and vertical_lines >= 2:
        court_prob = max(dark_ratio, red_ratio) * 0.8
        sport_probs['Basketball'] = min(1.0, court_prob)
        env_results['detected_environments'].append('basketball court')
        env_results['surface_type'] = 'court'

    env_results['sport_probabilities'] = sport_probs

    # Tính mức độ tin cậy tổng thể
    if sport_probs:
        max_prob = max(sport_probs.values())
        env_results['confidence'] = max_prob

    return env_results

# Replace the existing detect_human_pose function 
def detect_human_pose(img_data, conf_threshold=0.15, main_subject_box=None, sport_type=None):
    """
    Detect human pose and analyze pose patterns for sport identification
    
    Args:
        img_data: Dict containing image data
        conf_threshold: Confidence threshold for keypoint detection
        main_subject_box: Optional bounding box of main subject
        sport_type: Optional initially detected sport type to refine analysis
        
    Returns:
        Dict containing pose information and sport identification
    """
    if not POSE_MODEL_AVAILABLE:
        return {"poses": []}

    # Create result structure
    pose_results = {
        "poses": [],
        "pose_pattern": None,
        "identified_sport": None,
        "pattern_confidence": 0.0,
        "keypoint_names": {
            0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
            5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
            9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
            13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
        }
    }
    
    detected_equipment = []
    if 'detections' in img_data and 'classes' in img_data['detections']:
        detected_equipment = [cls for cls in img_data['detections']['classes'] 
                            if cls in ['sports ball', 'tennis racket', 'baseball bat', 
                                     'baseball glove', 'bicycle', 'soccer ball', 
                                     'football', 'rugby ball', 'golf club']]
        
    # Load model (only load once)
    if not hasattr(detect_human_pose, 'model'):
        print("Loading YOLOv8-Pose model...")
        detect_human_pose.model = YOLO('yolov8x-pose-p6.pt')
        print("YOLOv8-Pose model loaded successfully")

    # Predict on image
    results = detect_human_pose.model(img_data['resized_array'])

    for result in results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.data
            for person_id, person_keypoints in enumerate(keypoints):
                # Create pose entry for each person
                person_pose = {
                    "person_id": person_id,
                    "keypoints": [],
                    "bbox": None,
                    "pose_analysis": None
                }

                # Extract keypoints with confidence above threshold
                valid_keypoints = {}
                for kp_id, kp in enumerate(person_keypoints):
                    x, y, conf = kp.tolist()
                    if conf >= conf_threshold:
                        valid_keypoints[kp_id] = (float(x), float(y))
                        person_pose["keypoints"].append({
                            "id": kp_id,
                            "name": pose_results["keypoint_names"].get(kp_id, f"kp_{kp_id}"),
                            "x": float(x),
                            "y": float(y),
                            "confidence": float(conf)
                        })

                # Calculate bounding box from keypoints
                if person_pose["keypoints"]:
                    kp_coords = [(kp["x"], kp["y"]) for kp in person_pose["keypoints"]]
                    x_min = min([x for x, _ in kp_coords])
                    y_min = min([y for _, y in kp_coords])
                    x_max = max([x for x, _ in kp_coords])
                    y_max = max([y for _, y in kp_coords])
                    person_pose["bbox"] = [x_min, y_min, x_max, y_max]

                    # Add pose pattern analysis if we have enough keypoints
                    if len(valid_keypoints) >= 10:
                        skeleton = [
                            (5, 7), (7, 9),   # Left arm
                            (6, 8), (8, 10),  # Right arm
                            (5, 6),          # Shoulders
                            (5, 11), (6, 12), # Body
                            (11, 13), (13, 15), # Left leg
                            (12, 14), (14, 16), # Right leg
                            (11, 12)         # Hips
                        ]
                        
                        pose_analysis = analyze_pose_patterns(
                            valid_keypoints,
                            skeleton,
                            sport_type,
                            detected_equipment,
                            img_data
                        )
                        person_pose["pose_analysis"] = pose_analysis

                        # Update overall results if better confidence found
                        if pose_analysis["confidence"] > pose_results["pattern_confidence"]:
                            pose_results["pose_pattern"] = pose_analysis["pose_pattern"]
                            pose_results["identified_sport"] = pose_analysis["identified_sport"]
                            pose_results["pattern_confidence"] = pose_analysis["confidence"]

                # Add to results if valid
                if person_pose["keypoints"]:
                    pose_results["poses"].append(person_pose)

    print(f"DEBUG - pose_results has {len(pose_results.get('poses', []))} poses")
    return pose_results

def segment_main_subject(img, yolo_seg, main_subject_box):
    """
    Phân đoạn main subject bằng cách so sánh box với masks từ YOLOv8-seg

    Args:
        img: Ảnh cần phân tích (numpy array)
        yolo_seg: Model YOLOv8-seg đã load
        main_subject_box: Bounding box của main subject [x1, y1, x2, y2]

    Returns:
        mask: Numpy array chứa mask của main subject hoặc None nếu không tìm thấy
    """
    # Chạy YOLOv8-seg để lấy masks
    results = yolo_seg(img)

    best_mask = None
    best_iou = 0

    # Kiểm tra các kết quả
    for result in results:
        if hasattr(result, 'masks') and result.masks is not None:
            for i, mask in enumerate(result.masks.data):
                # Lấy bbox tương ứng với mask
                box = result.boxes.xyxy[i].cpu().numpy()

                # Tính IoU giữa box của mask và main_subject_box
                x1 = max(box[0], main_subject_box[0])
                y1 = max(box[1], main_subject_box[1])
                x2 = min(box[2], main_subject_box[2])
                y2 = min(box[3], main_subject_box[3])

                if x2 <= x1 or y2 <= y1:
                    continue

                intersection = (x2 - x1) * (y2 - y1)
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                subject_area = (main_subject_box[2] - main_subject_box[0]) * (main_subject_box[3] - main_subject_box[1])
                union = box_area + subject_area - intersection

                iou = intersection / union

                if iou > best_iou:
                    best_iou = iou
                    best_mask = mask.cpu().numpy()

    if best_mask is not None and best_iou > 0.5:  # Ngưỡng IoU để chấp nhận mask
        print(f"Tìm thấy mask cho main subject với IoU = {best_iou:.2f}")
        return best_mask

    print("Không tìm thấy mask phù hợp cho main subject")
    return None


def analyze_sports_composition(detections, analysis, img_data):
    """Analyze the composition with sports-specific context"""

    # Basic composition from existing analysis
    analysis["composition_analysis"] if "composition_analysis" in analysis else {}

    # Phân tích môi trường thể thao
    depth_map = None
    if 'depth_map' in analysis:
        depth_map = analysis['depth_map']

    env_analysis = analyze_sports_environment(img_data, depth_map)

    # Sports specific enhancements
    result = {
        'sport_type': 'Running',
        'framing_quality': 'Unknown',
        'recommended_crop': None,
        'action_focus': 'Unknown'
    }

    # MAPPING CHỈ CÁC ĐỐI TƯỢNG YOLO THỰC + CLIP BALL CLASSIFICATION
    sport_equipment = {
        # Đối tượng YOLO cơ bản
        'tennis racket': 'Tennis',
        'baseball bat': 'Baseball',
        'baseball glove': 'Baseball',
        'skateboard': 'Skateboarding',
        'surfboard': 'Surfing',
        'frisbee': 'Ultimate Frisbee',
        'skis': 'Alpine Skiing',
        'snowboard': 'Snowboarding',
        'bicycle': 'Cycling',
        'motorcycle': 'Motocross',
        'kite': 'Kite Sports',

        # Các loại bóng được CLIP phân loại từ 'sports ball'
        'soccer ball': 'Soccer',
        'basketball': 'Basketball',
        'volleyball': 'Volleyball',
        'tennis ball': 'Tennis',
        'baseball': 'Baseball',
        'golf ball': 'Golf',
        'american football': 'American Football',
        'rugby ball': 'Rugby',
        'ping pong ball': 'Table Tennis',
        'bowling ball': 'Bowling',
        'beach ball': 'Beach Sports'
    }

    # Cập nhật kết quả dựa trên môi trường
    detected_sport_from_env = None
    env_confidence = 0.0

    # Lấy môn thể thao có xác suất cao nhất từ phân tích môi trường
    if env_analysis['sport_probabilities']:
        env_sport, env_prob = max(env_analysis['sport_probabilities'].items(), key=lambda x: x[1])
        if env_prob > 0.8:  # Tăng ngưỡng tin cậy từ 0.5 lên 0.8 để giảm lỗi phân loại
            detected_sport_from_env = env_sport
            env_confidence = env_prob
            print(f"Phát hiện môn thể thao từ môi trường: {env_sport} (độ tin cậy: {env_prob:.2f})")
        else:
            print(f"Độ tin cậy phân tích môi trường quá thấp ({env_prob:.2f} < 0.8), bỏ qua kết quả: {env_sport}")

    detected_sport = None
    equipment_confidence = 0.0

    for cls in detections['classes']:
        if cls in sport_equipment:
            detected_sport = sport_equipment[cls]
            equipment_confidence = 0.7  # Độ tin cậy khi phát hiện từ dụng cụ
            break

    # Quyết định cuối cùng về môn thể thao
    if detected_sport and equipment_confidence > env_confidence:
        result['sport_type'] = detected_sport
    elif detected_sport_from_env:
        result['sport_type'] = detected_sport_from_env

    # Lưu thông tin phân tích môi trường
    result['environment_analysis'] = env_analysis

    framing_score = 0.0
    framing_details = {}

    sports_analysis_data = analysis.get('sports_analysis', {})

    if "key_subjects" in sports_analysis_data and sports_analysis_data['key_subjects']:
        print("=== Analyzing Framing Quality ===")

        # THÊM: Phát hiện nhóm vận động viên
        people_subjects = [s for s in sports_analysis_data['key_subjects'] if s['class'] == 'person']
        total_athletes = detections.get('athletes', 0)

        print(f"DEBUG - Total athletes detected: {total_athletes}")
        print(f"DEBUG - People in key_subjects: {len(people_subjects)}")

        # LOGIC MỚI: Phát hiện nhóm đông người (TIÊU CHÍ NGHIÊM NGẶT HƠN)
        is_group_scene = False

        # Điều kiện 1: Có nhiều người
        if total_athletes >= 3 and len(people_subjects) >= 2:
            print("DEBUG - Condition 1 met: Multiple people detected")

            # Điều kiện 2: Kích thước đối tượng chính > 35% = có thể là nhóm
            main_subject_temp = sports_analysis_data['key_subjects'][0]
            temp_box = main_subject_temp['box']
            temp_area = (temp_box[2] - temp_box[0]) * (temp_box[3] - temp_box[1])
            temp_ratio = temp_area / (img_data['resized_array'].shape[0] * img_data['resized_array'].shape[1])

            print(f"DEBUG - Main subject size ratio: {temp_ratio:.3f}")

            if temp_ratio > 0.35:  # Nếu đối tượng chính chiếm > 35% ảnh
                is_group_scene = True
                print("DEBUG - Condition 2 met: Large subject size suggests group")

            # Điều kiện 3: Kiểm tra khoảng cách (CHỈ NẾU CHƯA XÁC ĐỊNH ĐƯỢC)
            if not is_group_scene and len(people_subjects) >= 3:
                # Kiểm tra xem các người có gần nhau không
                positions = [s['position'] for s in people_subjects[:6]]  # Lấy tối đa 6 người đầu

                # Tính khoảng cách trung bình giữa các người
                distances = []
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        dist = np.sqrt((positions[i][0] - positions[j][0]) ** 2 +
                                       (positions[i][1] - positions[j][1]) ** 2)
                        distances.append(dist)

                if distances:
                    avg_distance = np.mean(distances)
                    print(f"DEBUG - Average distance between people: {avg_distance:.3f}")

                    # Nếu khoảng cách trung bình < 0.3 (30% ảnh) = nhóm sát nhau
                    if avg_distance < 0.3:
                        is_group_scene = True
                        print("DEBUG - Detected GROUP SCENE (crowded athletes)")

        if is_group_scene:
            # PHÂN TÍCH NHÓM: Lấy bounding box bao quanh toàn bộ nhóm
            all_boxes = [s['box'] for s in people_subjects[:8]]  # Tối đa 8 người

            # Tính group bounding box
            min_x = min([box[0] for box in all_boxes])
            min_y = min([box[1] for box in all_boxes])
            max_x = max([box[2] for box in all_boxes])
            max_y = max([box[3] for box in all_boxes])

            group_box = [min_x, min_y, max_x, max_y]
            group_pos = [(min_x + max_x) / 2 / img_data['resized_array'].shape[1],
                         (min_y + max_y) / 2 / img_data['resized_array'].shape[0]]

            main_subject = {
                'box': group_box,
                'position': group_pos,
                'class': 'group',
                'is_group': True
            }

            print(f"DEBUG - Group position: {group_pos}")
            print(f"DEBUG - Group box: {group_box}")
        else:
            # PHÂN TÍCH ĐƠN LẺ: Lấy đối tượng chính như cũ
            main_subject = sports_analysis_data['key_subjects'][0]
            main_subject['is_group'] = False

        main_pos = main_subject['position']
        main_box = main_subject['box']

        print(f"Main subject position: {main_pos}")
        print(f"Main subject class: {main_subject['class']}")

        # 1. PHÂN TÍCH VỊ TRÍ THEO RULE OF THIRDS
        # Các điểm vàng theo quy tắc 1/3
        rule_of_thirds_points = [
            (1 / 3, 1 / 3), (2 / 3, 1 / 3),  # Top left, top right
            (1 / 3, 2 / 3), (2 / 3, 2 / 3)  # Bottom left, bottom right
        ]

        # Tìm điểm gần nhất với rule of thirds
        min_dist_to_thirds = float('inf')

        for third_point in rule_of_thirds_points:
            dist = np.sqrt((main_pos[0] - third_point[0]) ** 2 + (main_pos[1] - third_point[1]) ** 2)
            if dist < min_dist_to_thirds:
                min_dist_to_thirds = dist

        # Điểm cho rule of thirds (càng gần càng cao)
        thirds_score = max(0, 1 - (min_dist_to_thirds / 0.3))  # Ngưỡng 30% khoảng cách
        print(f"Rule of thirds score: {thirds_score:.3f} (distance: {min_dist_to_thirds:.3f})")

        # 2. PHÂN TÍCH VỊ TRÍ TRUNG TÂM
        center_dist = np.sqrt((main_pos[0] - 0.5) ** 2 + (main_pos[1] - 0.5) ** 2)
        center_score = max(0, 1 - (center_dist / 0.4))  # Ngưỡng 40% từ trung tâm
        print(f"Center placement score: {center_score:.3f} (distance: {center_dist:.3f})")

        # 3. PHÂN TÍCH KÍCH THƯỚC ĐỐI TƯỢNG CHÍNH
        img_height = img_data['resized_array'].shape[0]
        img_width = img_data['resized_array'].shape[1]

        subject_width = main_box[2] - main_box[0]
        subject_height = main_box[3] - main_box[1]
        subject_area = subject_width * subject_height
        img_area = img_width * img_height

        size_ratio = subject_area / img_area
        print(f"Subject size ratio: {size_ratio:.3f}")

        # ĐIỀU CHỈNH: Tiêu chuẩn kích thước NGHIÊM NGẶT cho nhóm
        if main_subject.get('is_group', False):
            # Nhóm người: Áp dụng PENALTY cho kích thước quá lớn
            if 0.25 <= size_ratio <= 0.45:
                size_score = 1.0  # Kích thước lý tưởng cho nhóm
            elif 0.45 < size_ratio <= 0.60:
                size_score = 0.6  # PENALTY: Nhóm hơi lớn
            elif 0.60 < size_ratio <= 0.75:
                size_score = 0.3  # PENALTY MẠNH: Nhóm quá lớn
            elif size_ratio > 0.75:
                size_score = 0.1  # PENALTY RẤT MẠNH: Nhóm chiếm gần hết ảnh
            elif 0.15 <= size_ratio < 0.25:
                size_score = 0.7  # Nhóm hơi nhỏ
            else:
                size_score = 0.4  # Nhóm quá nhỏ

            print(f"DEBUG - Applied GROUP size scoring: {size_score:.3f} (ratio: {size_ratio:.3f})")
        else:
            # Đơn lẻ: kích thước lý tưởng nhỏ hơn (15-40%)
            if 0.15 <= size_ratio <= 0.40:
                size_score = 1.0
            elif 0.10 <= size_ratio < 0.15:
                size_score = 0.8
            elif 0.40 < size_ratio <= 0.60:
                size_score = 0.7
            elif 0.05 <= size_ratio < 0.10:
                size_score = 0.5
            else:
                size_score = 0.3
            print("DEBUG - Applied INDIVIDUAL size scoring")

        print(f"Size score: {size_score:.3f}")

        # 4. PHÂN TÍCH KHÔNG GIAN XUNG QUANH (BREATHING ROOM)
        # Kiểm tra đối tượng có quá gần viền không
        margin_left = main_pos[0]
        margin_right = 1 - main_pos[0]
        margin_top = main_pos[1]
        margin_bottom = 1 - main_pos[1]

        min_margin = min(margin_left, margin_right, margin_top, margin_bottom)

        # ĐIỀU CHỈNH: Tiêu chuẩn margin NGHIÊM NGẶT cho nhóm
        if main_subject.get('is_group', False):
            # Nhóm người: PENALTY cho margin quá lớn (tức là nhóm quá nhỏ hoặc ở giữa)
            if 0.05 <= min_margin <= 0.15:
                breathing_score = 1.0  # Margin lý tưởng cho nhóm
            elif 0.15 < min_margin <= 0.25:
                breathing_score = 0.7  # Hơi nhiều không gian trống
            elif 0.25 < min_margin <= 0.35:
                breathing_score = 0.4  # PENALTY: Quá nhiều không gian trống
            elif min_margin > 0.35:
                breathing_score = 0.2  # PENALTY MẠNH: Nhóm quá nhỏ so với ảnh
            elif 0.02 <= min_margin < 0.05:
                breathing_score = 0.8  # Hơi sát viền nhưng ok cho nhóm
            else:
                breathing_score = 0.3  # Quá sát viền

            print(f"DEBUG - Applied GROUP breathing room scoring: {breathing_score:.3f} (margin: {min_margin:.3f})")
        else:
            # Đơn lẻ: yêu cầu margin cao hơn
            if min_margin >= 0.15:
                breathing_score = 1.0
            elif min_margin >= 0.10:
                breathing_score = 0.8
            elif min_margin >= 0.05:
                breathing_score = 0.5
            else:
                breathing_score = 0.2
            print("DEBUG - Applied INDIVIDUAL breathing room scoring")

        print(f"Breathing room score: {breathing_score:.3f} (min margin: {min_margin:.3f})")

        # 5. PHÂN TÍCH ĐỐI TƯỢNG PHỤ
        secondary_objects_score = 1.0

        if len(sports_analysis_data['key_subjects']) > 1:
            # Kiểm tra đối tượng phụ có che khuất đối tượng chính không
            overlap_penalty = 0

            for i, secondary_subject in enumerate(sports_analysis_data['key_subjects'][1:4]):  # Chỉ xét 3 đối tượng đầu
                secondary_subject['position']
                sec_box = secondary_subject['box']

                # Tính overlap với đối tượng chính
                x1_main, y1_main, x2_main, y2_main = main_box
                x1_sec, y1_sec, x2_sec, y2_sec = sec_box

                # Tính intersection
                x_left = max(x1_main, x1_sec)
                y_top = max(y1_main, y1_sec)
                x_right = min(x2_main, x2_sec)
                y_bottom = min(y2_main, y2_sec)

                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    main_area = (x2_main - x1_main) * (y2_main - y1_main)
                    overlap_ratio = intersection / main_area

                    if overlap_ratio > 0.2:  # Nếu che khuất > 20%
                        overlap_penalty += overlap_ratio * 0.3
                        print(f"Overlap detected with secondary object {i + 1}: {overlap_ratio:.3f}")

            secondary_objects_score = max(0.2, 1.0 - overlap_penalty)

        print(f"Secondary objects score: {secondary_objects_score:.3f}")

        # 6. ĐIỀU CHỈNH THEO LOẠI THỂ THAO VÀ NHÓM
        sport_bonus = 1.0
        sport_type = result.get('sport_type', 'Unknown').lower()

        if main_subject.get('is_group', False):
            # NHÓM: Ưu tiên kích thước và breathing room hơn vị trí
            if sport_type in ['track', 'running', 'marathon']:
                sport_bonus = 1.0 + (size_score * 0.15) + (breathing_score * 0.10)
            elif sport_type in ['soccer', 'football', 'basketball']:
                sport_bonus = 1.0 + (size_score * 0.10) + (thirds_score * 0.10)
            else:
                sport_bonus = 1.0 + (size_score * 0.10)
            print("DEBUG - Applied GROUP sport bonus")
        else:
            # ĐƠN LẺ: Logic cũ
            if sport_type in ['soccer', 'football', 'basketball']:
                sport_bonus = 1.0 + (thirds_score * 0.2)
            elif sport_type in ['tennis', 'golf', 'individual']:
                sport_bonus = 1.0 + (center_score * 0.2)
            elif sport_type in ['track', 'running', 'swimming']:
                sport_bonus = 1.0 + (size_score * 0.1) + (breathing_score * 0.1)
            print("DEBUG - Applied INDIVIDUAL sport bonus")

        print(f"Sport bonus: {sport_bonus:.3f}")

        # 7. TÍNH ĐIỂM TỔNG HỢP
        # Trọng số cho từng yếu tố
        position_weight = 0.35  # Rule of thirds hoặc center
        size_weight = 0.25
        breathing_weight = 0.20
        secondary_weight = 0.20

        # Chọn điểm cao hơn giữa rule of thirds và center
        position_score = max(thirds_score, center_score)

        framing_score = (
                                position_score * position_weight +
                                size_score * size_weight +
                                breathing_score * breathing_weight +
                                secondary_objects_score * secondary_weight
                        ) * sport_bonus

        # 8. THÊM PENALTY ĐẶC BIỆT CHO NHÓM ĐÔNG NGƯỜI
        group_penalty = 1.0
        if main_subject.get('is_group', False):
            # Penalty dựa trên số lượng người và mật độ
            if total_athletes >= 8:
                group_penalty = 0.85  # 15% penalty cho nhóm rất đông
            elif total_athletes >= 5:
                group_penalty = 0.90  # 10% penalty cho nhóm đông

            # Penalty thêm nếu kích thước + margin đều cao (= framing kém)
            if size_ratio > 0.5 and min_margin > 0.3:
                group_penalty *= 0.8  # 20% penalty thêm
                print("DEBUG - Applied DOUBLE PENALTY for large group with too much space")

            print(f"DEBUG - Group penalty applied: {group_penalty:.3f}")

        # 9. TÍNH ĐIỂM TỔNG HỢP VỚI PENALTY
        # Trọng số cho từng yếu tố
        position_weight = 0.25  # Giảm từ 0.35 xuống 0.25 cho nhóm
        size_weight = 0.35  # Tăng từ 0.25 lên 0.35 (quan trọng hơn)
        breathing_weight = 0.25  # Tăng từ 0.20 lên 0.25
        secondary_weight = 0.15  # Giảm từ 0.20 xuống 0.15

        # Chọn điểm cao hơn giữa rule of thirds và center
        position_score = max(thirds_score, center_score)

        framing_score = (
                                position_score * position_weight +
                                size_score * size_weight +
                                breathing_score * breathing_weight +
                                secondary_objects_score * secondary_weight
                        ) * sport_bonus * group_penalty  # THÊM GROUP PENALTY

        framing_score = min(1.0, framing_score)  # Cap ở 1.0

        print(f"Final framing score: {framing_score:.3f}")

        # 10. PHÂN LOẠI CHẤT LƯỢNG (NGHIÊM NGẶT HƠN CHO NHÓM)
        if main_subject.get('is_group', False):
            # Tiêu chuẩn nghiêm ngặt hơn cho nhóm
            if framing_score >= 0.90:
                framing_quality = 'Excellent'
            elif framing_score >= 0.75:
                framing_quality = 'Very Good'
            elif framing_score >= 0.60:
                framing_quality = 'Good'
            elif framing_score >= 0.45:
                framing_quality = 'Fair'
            else:
                framing_quality = 'Could be improved'
            print("DEBUG - Applied GROUP quality standards")
        else:
            # Tiêu chuẩn bình thường cho đơn lẻ
            if framing_score >= 0.85:
                framing_quality = 'Excellent'
            elif framing_score >= 0.70:
                framing_quality = 'Very Good'
            elif framing_score >= 0.55:
                framing_quality = 'Good'
            elif framing_score >= 0.40:
                framing_quality = 'Fair'
            else:
                framing_quality = 'Could be improved'
            print("DEBUG - Applied INDIVIDUAL quality standards")

        result['framing_quality'] = framing_quality

        # Lưu chi tiết phân tích
        framing_details = {
            'overall_score': framing_score,
            'position_score': position_score,
            'size_score': size_score,
            'breathing_score': breathing_score,
            'secondary_objects_score': secondary_objects_score,
            'sport_bonus': sport_bonus,
            'main_subject_position': main_pos,
            'size_ratio': size_ratio,
            'min_margin': min_margin,
            'rule_of_thirds_distance': min_dist_to_thirds,
            'center_distance': center_dist
        }

        print(f"=== Framing Quality: {framing_quality} ({framing_score:.3f}) ===")

    else:
        result['framing_quality'] = 'Cannot analyze - no subjects detected'
        framing_details = {'error': 'No key subjects found'}
        print("Cannot analyze framing quality - no key subjects detected")

    # Lưu chi tiết vào kết quả
    result['framing_analysis'] = framing_details

    # Recommend crop if needed
    sports_analysis_data = analysis.get('sports_analysis', {})

    if "key_subjects" in sports_analysis_data and sports_analysis_data['key_subjects']:
        main_subject = sports_analysis_data['key_subjects'][0]
        x_pos = main_subject['position'][0]
        y_pos = main_subject['position'][1]

        # If subject is too far from ideal positions, suggest crop
        if not (0.3 < x_pos < 0.7 or 0.3 < y_pos < 0.7):
            # Calculate ideal center point
            if x_pos < 0.33:
                ideal_x = 0.33
            elif x_pos > 0.67:
                ideal_x = 0.67
            else:
                ideal_x = 0.5

            if y_pos < 0.33:
                ideal_y = 0.33
            elif y_pos > 0.67:
                ideal_y = 0.67
            else:
                ideal_y = 0.5

            # Calculate shift needed
            shift_x = ideal_x - x_pos
            shift_y = ideal_y - y_pos

            result['recommended_crop'] = {
                'shift_x': shift_x,
                'shift_y': shift_y
            }

    # Evaluate action focus
    if "action_quality" in analysis:
        result['action_focus'] = analysis['action_quality']

    return result


# Hàm phân tích biểu cảm nâng cao kết hợp DeepFace và phân tích ngữ cảnh

def analyze_facial_expression_advanced(detections, img_data, depth_map=None, sports_analysis=None):
    """Phân tích biểu cảm khuôn mặt nâng cao với OpenCV và HSEmotion, tập trung vào đối tượng chính"""
    try:
        print("Starting advanced facial expression analysis...")
        import cv2
        import numpy as np
        import os
        import traceback

        # Thiết lập để giảm lỗi TF/protobuf
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

        image = img_data['resized_array']
        image.shape[0] * image.shape[1]

        # Directory để lưu debug images
        debug_dir = "face_debug"
        os.makedirs(debug_dir, exist_ok=True)

        # Default results with debug info
        expression_results = {
            'has_faces': False,
            'expressions': [],
            'dominant_emotion': "unknown",
            'emotion_intensity': 0,
            'emotional_value': 'Low',
            'debug_info': {'detected': False, 'reason': 'No face detected initially'}
        }

        # 1. IDENTIFY MAIN SUBJECT
        main_subject = None
        main_subject_idx = -1

        # A. Từ phân tích sports_analysis
        if sports_analysis and 'key_subjects' in sports_analysis and sports_analysis['key_subjects']:
            for idx, subject in enumerate(sports_analysis['key_subjects']):
                if subject['class'] == 'person':
                    main_subject = subject
                    main_subject_idx = idx
                    print(
                        f"Main subject identified from key_subjects (idx={idx}, prominence={subject['prominence']:.3f}, sharpness={subject.get('sharpness', 0):.3f})")
                    break

        # B. Nếu không tìm thấy từ key_subjects, tìm từ detections
        if main_subject is None and isinstance(detections, dict) and 'boxes' in detections:
            max_center_weight = 0

            # Kích thước ảnh
            height, width = image.shape[:2]
            center_x, center_y = width // 2, height // 2

            for i, cls in enumerate(detections['classes']):
                if cls == 'person':
                    box = detections['boxes'][i]
                    x1, y1, x2, y2 = box

                    # Tính diện tích
                    area = (x2 - x1) * (y2 - y1)

                    # Tính vị trí trung tâm và độ gần với trung tâm ảnh
                    obj_center_x = (x1 + x2) // 2
                    obj_center_y = (y1 + y2) // 2

                    # Khoảng cách đến trung tâm (chuẩn hóa)
                    dist_to_center = np.sqrt(((obj_center_x - center_x) / width) ** 2 +
                                             ((obj_center_y - center_y) / height) ** 2)

                    # Trọng số kết hợp (diện tích + vị trí trung tâm)
                    center_weight = area * (1 - min(1.0, dist_to_center))

                    # Lưu giá trị cao nhất
                    if center_weight > max_center_weight:
                        max_center_weight = center_weight
                        main_subject = {
                            'box': box,
                            'class': cls,
                            'prominence': center_weight
                        }
                        main_subject_idx = i

            if main_subject:
                print(
                    f"Main subject identified from detections (idx={main_subject_idx}, prominence={main_subject['prominence']:.3f})")

        # Nếu không tìm thấy đối tượng chính
        if main_subject is None:
            print("Could not identify a main subject in the image")
            expression_results['debug_info']['reason'] = "No main subject detected"
            return expression_results

        # 2. EXTRACT MAIN SUBJECT REGION
        x1, y1, x2, y2 = main_subject['box']

        # Lưu ảnh đối tượng chính để debug
        subject_img = image[max(0, y1):min(y2, image.shape[0]),
                      max(0, x1):min(x2, image.shape[1])]
        subject_path = f"{debug_dir}/main_subject.jpg"
        cv2.imwrite(subject_path, cv2.cvtColor(subject_img, cv2.COLOR_RGB2BGR))

        # 3. PHÁT HIỆN KHUÔN MẶT TRONG VÙNG ĐẦU CỦA ĐỐI TƯỢNG CHÍNH
        h, w = subject_img.shape[:2]
        head_height = int(h * 0.4)  # Vùng đầu chiếm 40% trên của đối tượng
        head_region = subject_img[0:head_height, 0:w]

        # Lưu vùng đầu để debug
        head_path = f"{debug_dir}/head_region.jpg"
        cv2.imwrite(head_path, cv2.cvtColor(head_region, cv2.COLOR_RGB2BGR))

        # Phát hiện khuôn mặt CHỈ trong vùng đầu của đối tượng chính
        faces = detect_faces_improved(head_region)

        # Kiểm tra nếu tìm thấy khuôn mặt
        face_found = len(faces) > 0
        face_img = None
        fx, fy, fw, fh = 0, 0, 0, 0

        if face_found:
            print(f"Found {len(faces)} faces in main subject's head region")

            # Chọn khuôn mặt tốt nhất nếu có nhiều khuôn mặt
            if len(faces) > 1:
                best_face = select_best_face(faces, head_region)
            else:
                best_face = faces[0]

            hx, hy, hw, hh = best_face

            # Trích xuất khuôn mặt và thêm padding
            padding = max(10, int(hw * 0.1))  # Padding tỷ lệ với kích thước khuôn mặt
            face_x1 = max(0, hx - padding)
            face_y1 = max(0, hy - padding)
            face_x2 = min(head_region.shape[1], hx + hw + padding)
            face_y2 = min(head_region.shape[0], hy + hh + padding)

            face_img = head_region[face_y1:face_y2, face_x1:face_x2]

            # Lưu khuôn mặt để debug
            best_face_path = f"{debug_dir}/best_face.jpg"
            cv2.imwrite(best_face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            print(f"Face dimensions: {face_img.shape[1]}x{face_img.shape[0]}")

            # Điều chỉnh tọa độ khuôn mặt về không gian hình ảnh gốc
            fx = hx + x1
            fy = hy + y1  # Khuôn mặt đã nằm trong vùng head_region

        if not face_found or face_img is None or face_img.size == 0:
            print("No valid face detected in the main subject's head region")
            expression_results['debug_info']['reason'] = "No valid face detected in the main subject"
            return expression_results

        # 4. KIỂM TRA TÍNH HỢP LỆ CỦA KHUÔN MẶT
        is_valid_face, reason = verify_face(face_img)
        if not is_valid_face:
            print(f"Face verification failed: {reason}")
            expression_results['debug_info']['reason'] = f"Invalid face: {reason}"
            return expression_results

        # 5. PHÂN TÍCH BIỂU CẢM SỬ DỤNG MODEL DNN
        try:
            print("Phân tích cảm xúc với DNN Model...")
            import os
            import urllib.request

            # Tạo thư mục cho model
            model_dir = "emotion_models"
            os.makedirs(model_dir, exist_ok=True)

            # Đường dẫn tới model và proto
            prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            prototxt_path = os.path.join(model_dir, "deploy.prototxt")

            # URL model GitHub chứa mô hình emotion recognition onnx đơn giản
            model_url = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-7.onnx"
            model_path = os.path.join(model_dir, "emotion-ferplus.onnx")

            # Tải model nếu chưa có
            if not os.path.exists(prototxt_path):
                print("Đang tải prototxt...")
                urllib.request.urlretrieve(prototxt_url, prototxt_path)

            if not os.path.exists(model_path):
                print(f"Đang tải emotion model từ GitHub...")
                urllib.request.urlretrieve(model_url, model_path)

            # Chuẩn bị ảnh cho model
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            face_resized = cv2.resize(face_gray, (64, 64))

            # Tiền xử lý ảnh - cân bằng histogram để tăng độ tương phản
            face_equalized = cv2.equalizeHist(face_resized)

            # Chuẩn hóa ảnh (giá trị pixel từ 0-1)
            tensor = face_equalized.reshape(1, 1, 64, 64).astype(np.float32)

            # Tải model và dự đoán
            print("Tải model DNN...")
            net = cv2.dnn.readNetFromONNX(model_path)
            net.setInput(tensor)
            output = net.forward()

            # Tính xác suất với softmax
            def softmax(x):
                exp_x = np.exp(x - np.max(x))
                return exp_x / exp_x.sum()

            probabilities = softmax(output[0])

            # Danh sách cảm xúc theo thứ tự của model FER+
            emotions = ['neutral', 'happy', 'surprise', 'sad', 'angry', 'disgust', 'fear']

            # Tạo từ điển điểm số
            emotion_scores_dict = {emotion: float(prob) for emotion, prob in zip(emotions, probabilities)}
            # Thêm contempt cho tương thích với code gốc
            emotion_scores_dict['contempt'] = 0.01
            emotion_scores_dict['focus'] = 0.01

            # Xác định cảm xúc chính
            dominant_emotion = max(emotion_scores_dict, key=emotion_scores_dict.get)
            dominant_score = emotion_scores_dict[dominant_emotion]

            print(f"DNN phát hiện cảm xúc: {dominant_emotion}")
            print(f"Điểm số cảm xúc: {emotion_scores_dict}")

            # Tính cường độ cảm xúc
            emotion_intensity = min(0.95, max(0.2, dominant_score))

            # Tạo ảnh debug
            debug_img = face_img.copy()

            # Hiển thị cảm xúc chính
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(debug_img, f"{dominant_emotion.upper()}: {dominant_score:.2f}",
                        (10, 30), font, 0.7, (0, 255, 0), 2)

            # Hiển thị điểm số cảm xúc
            y_pos = 60
            sorted_emotions = sorted(emotion_scores_dict.items(), key=lambda x: x[1], reverse=True)

            for emotion, score in sorted_emotions:
                if emotion == 'contempt':  # Bỏ qua contempt vì nó chỉ là giá trị giữ chỗ
                    continue

                bar_width = int(score * 200)
                bar_color = (0, 255, 0)  # Default: green

                # Màu riêng cho từng cảm xúc
                if emotion == 'happy':
                    bar_color = (0, 255, 255)  # Yellow
                elif emotion == 'sad':
                    bar_color = (255, 0, 0)  # Blue
                elif emotion == 'angry':
                    bar_color = (0, 0, 255)  # Red
                elif emotion == 'surprise':
                    bar_color = (255, 0, 255)  # Magenta

                cv2.rectangle(debug_img, (10, y_pos), (10 + bar_width, y_pos + 20),
                              bar_color, -1)
                cv2.putText(debug_img, f"{emotion}: {score:.2f}",
                            (15, y_pos + 15), font, 0.5, (255, 255, 255), 1)
                y_pos += 30

            # Hiển thị ảnh đã xử lý cho debugging
            h, w = face_equalized.shape
            display_face = cv2.resize(face_equalized, (w * 2, h * 2))
            display_face = cv2.cvtColor(display_face, cv2.COLOR_GRAY2BGR)

            # Lưu ảnh debug
            emotion_debug_path = f"{debug_dir}/dnn_emotion.jpg"
            cv2.imwrite(emotion_debug_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
            print(f"Đã lưu ảnh debug tại: {emotion_debug_path}")

        except Exception as e:
            print(f"DNN analysis failed: {str(e)}")
            print(f"Chi tiết: {traceback.format_exc()}")

            # GIẢI PHÁP DỰ PHÒNG: PHÂN TÍCH LBP + HOG FEATURES
            try:
                print("Sử dụng phương pháp phân tích đặc trưng LBP và HOG...")

                # Chuyển ảnh sang grayscale nếu chưa
                if len(face_img.shape) == 3:
                    gray_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                else:
                    gray_img = face_img.copy()

                # Đảm bảo ảnh có kích thước chuẩn
                gray_img = cv2.resize(gray_img, (96, 96))

                # 1. Trích xuất đặc trưng LBP
                def extract_lbp_features(image, radius=1, n_points=8):
                    lbp = np.zeros_like(image)
                    for i in range(radius, image.shape[0] - radius):
                        for j in range(radius, image.shape[1] - radius):
                            center = image[i, j]
                            binary_pattern = []

                            # So sánh từng điểm lân cận
                            binary_pattern.append(1 if image[i - 1, j - 1] >= center else 0)
                            binary_pattern.append(1 if image[i - 1, j] >= center else 0)
                            binary_pattern.append(1 if image[i - 1, j + 1] >= center else 0)
                            binary_pattern.append(1 if image[i, j + 1] >= center else 0)
                            binary_pattern.append(1 if image[i + 1, j + 1] >= center else 0)
                            binary_pattern.append(1 if image[i + 1, j] >= center else 0)
                            binary_pattern.append(1 if image[i + 1, j - 1] >= center else 0)
                            binary_pattern.append(1 if image[i, j - 1] >= center else 0)

                            lbp_value = 0
                            for k, bit in enumerate(binary_pattern):
                                lbp_value += bit * (2 ** k)

                            lbp[i, j] = lbp_value

                    return lbp

                # 2. Trích xuất đặc trưng gradient (đơn giản hóa HOG)
                def extract_gradient_features(image):
                    # Tính gradient theo trục x và y
                    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
                    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

                    # Tính magnitude và góc
                    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
                    gradient_angle = np.arctan2(gradient_y, gradient_x) * 180 / np.pi

                    return gradient_magnitude, gradient_angle

                # Trích xuất LBP
                lbp_image = extract_lbp_features(gray_img)

                # Chia ảnh thành vùng (3x3)
                h, w = lbp_image.shape
                cell_h, cell_w = h // 3, w // 3

                # Tính histogram cho từng vùng
                lbp_features = []
                for i in range(3):
                    for j in range(3):
                        cell = lbp_image[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                        hist, _ = np.histogram(cell, bins=16, range=(0, 256))
                        lbp_features.extend(hist)

                # Chuẩn hóa đặc trưng
                lbp_features = np.array(lbp_features) / np.sum(lbp_features)

                # Trích xuất gradient
                gradient_mag, gradient_angle = extract_gradient_features(gray_img)

                # Tính các vùng mặt
                # Vùng mắt (1/3 trên)
                eyes_region = gray_img[:h // 3, :]
                # Vùng mũi (1/3 giữa)
                nose_region = gray_img[h // 3:2 * h // 3, :]
                # Vùng miệng (1/3 dưới)
                mouth_region = gray_img[2 * h // 3:, :]

                # Tính các đặc trưng thống kê
                regions = {
                    'eyes': eyes_region,
                    'nose': nose_region,
                    'mouth': mouth_region
                }

                region_stats = {}
                for name, region in regions.items():
                    region_stats[name] = {
                        'mean': np.mean(region) / 255.0,
                        'std': np.std(region) / 255.0,
                        'gradient_mean': np.mean(gradient_mag[0:h // 3, :] if name == 'eyes' else
                                                 gradient_mag[h // 3:2 * h // 3, :] if name == 'nose' else
                                                 gradient_mag[2 * h // 3:, :]) / 255.0
                    }

                # Hệ số cho các đặc trưng
                emotion_scores_dict = {
                    'neutral': 0.2,
                    'happy': 0.1,
                    'sad': 0.1,
                    'surprise': 0.1,
                    'angry': 0.1,
                    'fear': 0.1,
                    'disgust': 0.1,
                    'contempt': 0.1,
                    'focus': 0.10  # Thêm cảm xúc focus
                }

                # Phân tích đặc trưng LBP
                # LBP patterns đặc trưng cho happy thường có nhiều điểm sáng (do nụ cười)
                lbp_bright_pattern = np.sum(lbp_features[np.arange(16) * 9 + 8])  # Kiểm tra mẫu bit sáng

                # Phân tích các vùng
                eyes_bright = region_stats['eyes']['mean']
                mouth_bright = region_stats['mouth']['mean']
                mouth_contrast = region_stats['mouth']['std']
                eyes_gradient = region_stats['eyes']['gradient_mean']
                mouth_gradient = region_stats['mouth']['gradient_mean']

                # Quy tắc phân loại
                # 1. Happy: Miệng sáng, độ tương phản cao (nụ cười)
                if mouth_bright > 0.5 and mouth_contrast > 0.16:
                    emotion_scores_dict['happy'] += 0.4
                    emotion_scores_dict['neutral'] -= 0.1

                # 2. Sad: Miệng tối, mắt tối
                if mouth_bright < 0.35 and eyes_bright < 0.4:
                    emotion_scores_dict['sad'] += 0.4
                    emotion_scores_dict['neutral'] -= 0.1

                # 3. Surprise: Gradient mắt và miệng cao (mắt mở to, miệng mở)
                if eyes_gradient > 0.12 and mouth_gradient > 0.15:
                    emotion_scores_dict['surprise'] += 0.4
                    emotion_scores_dict['fear'] += 0.2
                    emotion_scores_dict['neutral'] -= 0.2

                # 4. Angry: Eyes gradient cao, miệng tối
                if eyes_gradient > 0.15 and mouth_bright < 0.4:
                    emotion_scores_dict['angry'] += 0.3
                    emotion_scores_dict['disgust'] += 0.1

                # 5. Neutral: Ít biến đổi
                if abs(eyes_gradient - mouth_gradient) < 0.05 and 0.4 < mouth_bright < 0.6:
                    emotion_scores_dict['neutral'] += 0.3

                # 6. Tăng Happy dựa trên LBP pattern
                if lbp_bright_pattern > 0.15:
                    emotion_scores_dict['happy'] += 0.2
                    emotion_scores_dict['neutral'] -= 0.1

                # Đảm bảo không có giá trị âm
                emotion_scores_dict = {k: max(0.01, v) for k, v in emotion_scores_dict.items()}

                # Chuẩn hóa
                total = sum(emotion_scores_dict.values())
                emotion_scores_dict = {k: v / total for k, v in emotion_scores_dict.items()}

                # Tìm cảm xúc chính
                dominant_emotion = max(emotion_scores_dict, key=emotion_scores_dict.get)
                dominant_score = emotion_scores_dict[dominant_emotion]

                print(f"LBP/HOG phát hiện cảm xúc: {dominant_emotion}")
                print(f"Điểm số cảm xúc: {emotion_scores_dict}")

                # Tính cường độ cảm xúc
                emotion_intensity = min(0.95, max(0.2, dominant_score))

                # Tạo ảnh debug
                debug_img = face_img.copy()

                # Hiển thị cảm xúc chính
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(debug_img, f"{dominant_emotion.upper()}: {dominant_score:.2f}",
                            (10, 30), font, 0.7, (0, 255, 0), 2)

                # Chia vùng khuôn mặt để phân tích
                h, w = debug_img.shape[:2] if len(debug_img.shape) == 2 else debug_img.shape[:2]
                cv2.line(debug_img, (0, h // 3), (w, h // 3), (0, 255, 0), 1)
                cv2.line(debug_img, (0, 2 * h // 3), (w, 2 * h // 3), (0, 255, 0), 1)

                # Hiển thị các đặc trưng vùng
                y_pos = 50
                cv2.putText(debug_img, f"Eyes brightness: {eyes_bright:.2f}", (10, y_pos), font, 0.4, (255, 255, 255),
                            1)
                y_pos += 20
                cv2.putText(debug_img, f"Mouth brightness: {mouth_bright:.2f}", (10, y_pos), font, 0.4, (255, 255, 255),
                            1)
                y_pos += 20
                cv2.putText(debug_img, f"Mouth contrast: {mouth_contrast:.2f}", (10, y_pos), font, 0.4, (255, 255, 255),
                            1)

                # Hiển thị điểm số cảm xúc
                y_pos = h - 120
                for emotion, score in sorted(emotion_scores_dict.items(), key=lambda x: x[1], reverse=True):
                    bar_width = int(score * 150)
                    bar_color = (0, 255, 0)  # Default: green

                    if emotion == 'happy':
                        bar_color = (0, 255, 255)  # Yellow
                    elif emotion == 'sad':
                        bar_color = (255, 0, 0)  # Blue
                    elif emotion == 'angry':
                        bar_color = (0, 0, 255)  # Red

                    cv2.rectangle(debug_img, (10, y_pos), (10 + bar_width, y_pos + 15),
                                  bar_color, -1)
                    cv2.putText(debug_img, f"{emotion}: {score:.2f}",
                                (10 + bar_width + 5, y_pos + 12), font, 0.4, (255, 255, 255), 1)
                    y_pos += 20

                # Lưu ảnh debug
                emotion_debug_path = f"{debug_dir}/lbp_hog_emotion.jpg"
                cv2.imwrite(emotion_debug_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
                print(f"Đã lưu ảnh debug tại: {emotion_debug_path}")

            except Exception as e:
                print(f"LBP/HOG analysis failed: {str(e)}")
                print(f"Chi tiết: {traceback.format_exc()}")
                print("Falling back to basic emotion analysis...")

                # Phân tích đơn giản dựa trên độ tương phản và độ sáng
                gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
                brightness = np.mean(gray) / 255.0
                np.std(gray) / 128.0

                # Tính gradient (texture)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_mean = np.mean(np.sqrt(sobelx ** 2 + sobely ** 2)) / 128.0

                # Chia ảnh thành vùng
                h, w = gray.shape
                top = gray[:h // 3, :]  # Vùng mắt
                middle = gray[h // 3:2 * h // 3, :]  # Vùng mũi
                bottom = gray[2 * h // 3:, :]  # Vùng miệng

                # Tính các đặc trưng theo vùng
                np.std(top) / 128.0
                np.std(middle) / 128.0
                bottom_contrast = np.std(bottom) / 128.0

                # Khởi tạo điểm số cảm xúc
                emotion_scores_dict = {
                    'neutral': 0.10,
                    'happy': 0.10,
                    'surprise': 0.10,
                    'sad': 0.10,
                    'angry': 0.10,
                    'fear': 0.10,
                    'disgust': 0.10,
                    'focus': 0.10
                }

                # Phân tích độ sáng & tương phản
                if brightness > 0.6:  # Sáng -> vui/tích cực
                    emotion_scores_dict['happy'] += 0.15
                    emotion_scores_dict['neutral'] -= 0.05
                elif brightness < 0.4:  # Tối -> nghiêm trọng/tiêu cực
                    emotion_scores_dict['sad'] += 0.10
                    emotion_scores_dict['angry'] += 0.05
                    emotion_scores_dict['happy'] -= 0.05

                # Phân tích texture
                if gradient_mean > 0.25:  # Texture cao -> biểu cảm mạnh
                    emotion_scores_dict['surprise'] += 0.15
                    emotion_scores_dict['happy'] += 0.05
                    emotion_scores_dict['neutral'] -= 0.10

                # Phân tích vùng miệng
                if bottom_contrast > 0.20:
                    if brightness > 0.45:
                        emotion_scores_dict['happy'] += 0.25
                    else:
                        emotion_scores_dict['surprise'] += 0.20

                # Đảm bảo không có giá trị âm
                emotion_scores_dict = {k: max(0.01, v) for k, v in emotion_scores_dict.items()}

                # Chuẩn hóa
                total = sum(emotion_scores_dict.values())
                emotion_scores_dict = {k: v / total for k, v in emotion_scores_dict.items()}

                # Lấy cảm xúc chính
                dominant_emotion = max(emotion_scores_dict, key=emotion_scores_dict.get)
                dominant_score = emotion_scores_dict[dominant_emotion]

                # Tính cường độ
                other_scores = [v for k, v in emotion_scores_dict.items() if k != dominant_emotion]
                avg_other = sum(other_scores) / len(other_scores) if other_scores else 0
                emotion_intensity = max(0.2, min(0.95, 0.5 + (dominant_score - avg_other)))

                print(f"Basic detected emotion: {dominant_emotion}")

        # 6. PHÂN TÍCH NGỮ CẢNH THỂ THAO
        # Xác định loại thể thao
        sport_type = "Running"
        if isinstance(sports_analysis, dict):
            if "composition_analysis" in sports_analysis:
                sport_type = sports_analysis["composition_analysis"].get("sport_type", "Unknown")

        # Mức độ hành động
        action_level = 0
        if "action_analysis" in sports_analysis and "action_level" in sports_analysis["action_analysis"]:
            action_level = sports_analysis["action_analysis"]["action_level"]

        # Điều chỉnh biểu cảm dựa trên ngữ cảnh thể thao
        contextual_emotions = emotion_scores_dict.copy()

        # Các môn thể thao đối kháng
        combat_sports = ['Boxing', 'Wrestling', 'Martial Arts']
        team_sports = ['Soccer', 'Basketball', 'Baseball', 'Football', 'Ball Sport']
        track_sports = ['Running', 'Track', 'Sprint', 'Athletics']

        # Phát hiện thể thao điền kinh từ ảnh
        if any(name in str(sport_type).lower() for name in ['track', 'run', 'sprint', 'athlet']):
            print("Detected track and field sport from image")
            sport_type = 'Track and Field'

        # Điều chỉnh theo loại thể thao
        if sport_type in combat_sports:
            print(f"Adjusting emotions for combat sport: {sport_type}")
            # Tăng mạnh cảm xúc 'determination', 'angry', v.v. trong thể thao đối kháng
            contextual_emotions['angry'] = contextual_emotions.get('angry', 0) * 1.3
            contextual_emotions['fear'] = contextual_emotions.get('fear', 0) * 1.2
            emotion_intensity = min(0.95, emotion_intensity * 1.2)

        elif sport_type in team_sports:
            print(f"Adjusting emotions for team sport: {sport_type}")
            # Tăng cảm xúc vui mừng/thất vọng trong thể thao đồng đội
            if action_level > 0.5:  # Hành động cao
                contextual_emotions['happy'] = contextual_emotions.get('happy', 0) * 1.2
                contextual_emotions['surprise'] = contextual_emotions.get('surprise', 0) * 1.2

        # Điều chỉnh cho môn điền kinh với cảm xúc phù hợp hơn
        elif sport_type in track_sports or 'Track and Field' in sport_type:
            print(f"Adjusting emotions for track sport")
            # Giảm happy và tăng determination/effort
            contextual_emotions['happy'] = contextual_emotions.get('happy', 0) * 0.9  # Giảm happy

            # Nếu phát hiện angry hoặc neutral cao, đổi thành determination
            if contextual_emotions.get('angry', 0) > 0.2 or contextual_emotions.get('neutral', 0) > 0.3:
                # Tạo cảm xúc determination từ angry
                determination_score = contextual_emotions.get('angry', 0) * 1.8
                contextual_emotions['determination'] = determination_score

                # Nếu determination là cảm xúc chính
                if determination_score > max([v for k, v in contextual_emotions.items() if k != 'determination']):
                    dominant_emotion = 'determination'
                    print("Changed main emotion to 'determination' based on track sports context")

        # Chuyển đổi "neutral" thành "focus" trong bối cảnh thể thao
        if dominant_emotion == 'neutral' and emotion_intensity > 0.5:
            # Xem xét các điều kiện để xác định có phải đang tập trung hay không
            is_focus = False

            # Điều kiện 1: Đang trong môi trường thể thao và có mức độ hành động cao
            if action_level > 0.5:
                is_focus = True

            # Điều kiện 2: Trong môn thể thao đồng đội hoặc với bóng
            if sport_type in team_sports or "ball" in str(sport_type).lower():
                is_focus = True

            # Điều kiện 3: Kiểm tra cường độ cảm xúc và độ tin cậy
            if emotion_intensity > 0.7:
                is_focus = True

            # Nếu thỏa mãn điều kiện, thay đổi neutral thành focus
            if is_focus:
                dominant_emotion = 'focus'
                print("Changed 'neutral' to 'focus' based on sports context")

                # Cập nhật điểm số cảm xúc
                if 'neutral' in contextual_emotions:
                    contextual_emotions['focus'] = contextual_emotions.pop('neutral', dominant_score)
                else:
                    contextual_emotions['focus'] = dominant_score

                # Cũng cập nhật trong emotion_scores_dict gốc nếu cần
                if 'neutral' in emotion_scores_dict:
                    emotion_scores_dict['focus'] = emotion_scores_dict.pop('neutral', dominant_score)
                else:
                    emotion_scores_dict['focus'] = dominant_score

        # 7. PHÂN TÍCH MỨC ĐỘ CẢM XÚC
        emotional_value = 'Moderate'
        if emotion_intensity < 0.4:
            emotional_value = 'Low'
        elif emotion_intensity > 0.7:
            emotional_value = 'High'

        # Thêm kết quả vào expression_results
        expression_results['has_faces'] = True
        expression_results['dominant_emotion'] = dominant_emotion
        expression_results['emotion_intensity'] = float(emotion_intensity)
        expression_results['emotional_value'] = emotional_value
        expression_results['emotion_scores'] = emotion_scores_dict
        expression_results['contextual_scores'] = contextual_emotions
        expression_results['expressions'] = [{'emotion': dominant_emotion, 'score': float(dominant_score)}]
        expression_results['debug_info']['detected'] = True
        expression_results['face_path'] = best_face_path
        expression_results['face_coordinates'] = (fx, fy, fw, fh)
        expression_results['sport_context'] = sport_type
        expression_results['action_level'] = action_level

        # Thêm thông tin đối tượng chính
        expression_results['main_subject_idx'] = main_subject_idx
        expression_results['main_subject_box'] = main_subject['box']

        print(f"Advanced facial expression analysis successful: {face_found}")
        return expression_results

    except Exception as e:
        print(f"Error in advanced facial expression analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'has_faces': False, 'error': str(e), 'debug_info': {'reason': f"Error: {str(e)}"}}

def visualize_emotion_results(face_img, emotion_analysis):
    """Tạo hiển thị chuyên nghiệp cho phân tích biểu cảm khuôn mặt"""
    # Nếu không có phân tích cảm xúc, tạo hình ảnh thông báo lỗi
    if face_img is None or emotion_analysis is None or not emotion_analysis.get('has_faces', False):
        # Tạo hình ảnh trống với thông báo
        fig = plt.figure(figsize=(6, 4), dpi=100)

        # Thêm tiêu đề
        error_message = "NO FACE DETECTED"
        detail_message = "Cannot analyze facial expression"

        # Truy tìm lý do lỗi chi tiết hơn
        if emotion_analysis:
            if 'error' in emotion_analysis:
                detail_message = emotion_analysis['error']

            if 'debug_info' in emotion_analysis and 'reason' in emotion_analysis['debug_info']:
                detail_message = emotion_analysis['debug_info']['reason']

        # Tạo hình ảnh thông báo
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.6, error_message,
                fontsize=16, color='red', fontweight='bold',
                ha='center', va='center')
        ax.text(0.5, 0.4, detail_message,
                fontsize=12, ha='center', va='center',
                wrap=True)
        ax.axis('off')

        # Chuyển đổi figure thành ảnh
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Đóng figure để tránh memory leak
        plt.close(fig)

        return img_data

    # Đảm bảo kích thước hợp lý cho ảnh khuôn mặt
    h, w = face_img.shape[:2]
    display_width = 300
    display_height = int((h / w) * display_width) if w > 0 else 300

    # Điều chỉnh kích thước ảnh khuôn mặt cho phù hợp
    face_display = cv2.resize(face_img, (display_width, display_height),
                              interpolation=cv2.INTER_AREA)

    # Lấy thông tin cảm xúc
    emotion = emotion_analysis.get('dominant_emotion', 'unknown')
    intensity = emotion_analysis.get('emotion_intensity', 0)
    original_emotion = emotion_analysis.get('original_emotion', emotion)

    # Định nghĩa màu dựa trên loại cảm xúc (dạng RGB cho matplotlib)
    emotion_colors = {
        'happy': (0.0, 0.8, 0.2),  # Green
        'happiness': (0.0, 0.8, 0.2),  # Green
        'surprise': (1.0, 0.7, 0.0),  # Orange
        'sad': (0.0, 0.0, 0.8),  # Blue
        'sadness': (0.0, 0.0, 0.8),  # Blue
        'angry': (0.8, 0.0, 0.0),  # Red
        'anger': (0.8, 0.0, 0.0),  # Red
        'fear': (0.8, 0.0, 0.8),  # Purple
        'disgust': (0.5, 0.4, 0.0),  # Brown
        'neutral': (0.5, 0.5, 0.5),  # Gray
        'contempt': (0.6, 0.0, 0.6),  # Dark purple
        'unknown': (0.7, 0.7, 0.7)  # Light Gray
    }

    # Màu dựa trên cảm xúc phát hiện được
    color = emotion_colors.get(emotion.lower(), (0.7, 0.7, 0.7))

    # Tạo figure để hiển thị với kích thước hợp lý
    fig = plt.figure(figsize=(6, 8), dpi=100)

    # Thêm tiêu đề với màu theo cảm xúc
    emotion_title = f"Face: {emotion.upper()} ({intensity:.2f})"
    if original_emotion != emotion and original_emotion != "unknown":
        emotion_title += f"\nOriginal: {original_emotion.upper()}"

    fig.suptitle(emotion_title, fontsize=16, color=color, fontweight='bold')

    # Thiết lập layout - 2 dòng, 1 cột
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1], figure=fig)

    # Hiển thị ảnh khuôn mặt
    ax_face = fig.add_subplot(gs[0])
    ax_face.imshow(face_display)

    # Thêm viền màu xung quanh khuôn mặt
    border_width = 5
    for spine in ax_face.spines.values():
        spine.set_linewidth(border_width)
        spine.set_color(color)

    ax_face.set_xticks([])
    ax_face.set_yticks([])

    # Thêm biểu đồ cảm xúc nếu có điểm số
    scores = emotion_analysis.get('contextual_scores', emotion_analysis.get('emotion_scores', {}))
    if scores:
        ax_chart = fig.add_subplot(gs[1])

        # Sắp xếp cảm xúc theo điểm số
        emotions = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        values = [scores[e] for e in emotions]

        # Rút ngắn tên cảm xúc để hiển thị gọn hơn
        display_labels = [e[:3].upper() if len(e) > 3 else e.upper() for e in emotions]

        # Tạo màu cho từng cảm xúc
        bar_colors = [emotion_colors.get(e.lower(), (0.7, 0.7, 0.7)) for e in emotions]

        # Vẽ biểu đồ thanh ngang để dễ đọc hơn
        bars = ax_chart.barh(display_labels, values, color=bar_colors, alpha=0.7)

        # Thêm giá trị lên mỗi thanh
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax_chart.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                          f'{width:.2f}', ha='left', va='center', fontweight='bold')

        # Đặt giới hạn trục x từ 0 đến 1
        ax_chart.set_xlim(0, 1.0)

        # Tiêu đề nhỏ cho biểu đồ
        ax_chart.set_title('Emotion Scores', fontsize=12)

        # Thêm lưới để dễ đọc
        ax_chart.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Chuyển đổi figure thành ảnh
    fig.canvas.draw()
    img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Đóng figure để tránh memory leak
    plt.close(fig)

    return img_data


def visualize_sports_results(img_data, detections, depth_map, sports_analysis, action_analysis, composition_analysis,
                             facial_analysis=None, caption=None):
    """Create sports-specific visualization with enhanced main subject highlighting, emotion analysis and caption"""
    # Thêm debug để xác định ID của biến
    print(f"DEBUG D - ID của sports_analysis trong visualize: {id(sports_analysis)}")
    print(
        f"DEBUG - sports_analysis keys trong visualize: {sports_analysis.keys() if isinstance(sports_analysis, dict) else type(sports_analysis)}")

    img = np.array(img_data['resized']).copy()
    height, width = img.shape[:2]

    # Tìm đối tượng chính (người) từ key_subjects
    main_person = None
    main_person_idx = -1

    if "key_subjects" in sports_analysis and sports_analysis['key_subjects']:
        # Lấy chính xác đối tượng đầu tiên nếu là người
        if sports_analysis['key_subjects'][0]['class'] == 'person':
            main_person = sports_analysis['key_subjects'][0]
            main_person_idx = 0
        else:
            # Nếu không, tìm người đầu tiên trong danh sách
            for idx, subject in enumerate(sports_analysis['key_subjects']):
                if subject['class'] == 'person':
                    main_person = subject
                    main_person_idx = idx
                    break

    # Tạo visual cho detection với sharpness
    det_viz = img.copy()

    # Tạo mask để làm nổi bật đối tượng chính
    highlight_mask = np.zeros_like(img)
    main_obj_viz = img.copy()

    # Lấy điểm số sắc nét
    sharpness_scores = sports_analysis.get('sharpness_scores', [])

    # Draw bounding boxes
    for i, box in enumerate(detections['boxes']):
        x1, y1, x2, y2 = box
        label = detections['classes'][i]
        conf = detections['scores'][i]
        sharpness = sharpness_scores[i] if i < len(sharpness_scores) else 0
        object_id = i + 1

        # Determine if this is the main person
        is_main_person = False
        if main_person is not None and i == main_person_idx:
            is_main_person = True

        # Different colors for different classes
        if label == 'person':
            if is_main_person:
                # Highlight đối tượng chính với màu sáng và đậm hơn
                color = (0, 255, 255)  # Vàng cho người chính
                border_thickness = 4
                font_scale = 0.7

                # Tạo mask cho vùng đối tượng chính
                highlight_mask[y1:y2, x1:x2] = img[y1:y2, x1:x2]

                # Vẽ spotlight effect xung quanh người chính
                cv2.rectangle(main_obj_viz, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 255, 255), 8)

                # Thêm tên nhãn nổi bật hơn với ID
                cv2.putText(main_obj_viz, f"ID:{object_id} MAIN SUBJECT", (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                color = (0, 255, 0)  # Green for other people
                border_thickness = 2
                font_scale = 0.5
        elif 'ball' in label:
            color = (0, 0, 255)  # Red for balls
            border_thickness = 2
            font_scale = 0.5
        else:
            color = (255, 0, 0)  # Blue for other equipment
            border_thickness = 2
            font_scale = 0.5

        cv2.rectangle(det_viz, (x1, y1), (x2, y2), color, border_thickness)

        # Draw label with sharpness
        label_y = y1 - 10 if y1 > 20 else y1 + 20
        cv2.putText(det_viz, f"ID:{object_id} {label} {conf:.2f} S:{sharpness:.2f}", (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    # Tạo hiệu ứng làm mờ hình nền và làm nổi bật đối tượng chính
    if main_person is not None:
        # Blur background
        blurred_bg = cv2.GaussianBlur(img, (25, 25), 0)

        # Tạo mask cho vùng người chính
        person_mask = np.zeros((height, width), dtype=np.uint8)

        # Sử dụng mask chi tiết từ segmentation nếu có
        if 'main_subject_mask' in sports_analysis and sports_analysis['main_subject_mask'] is not None:
            main_mask = sports_analysis['main_subject_mask']

            # Resize mask nếu kích thước khác với ảnh
            if main_mask.shape[:2] != (height, width):
                main_mask = cv2.resize(main_mask, (width, height))

            # Chuyển mask về binary
            person_mask = (main_mask > 0.5).astype(np.uint8) * 255
        else:
            # Sử dụng bounding box nếu không có mask
            x1, y1, x2, y2 = main_person['box']
            person_mask[y1:y2, x1:x2] = 255

        # Thêm một border trơn mượt
        kernel = np.ones((15, 15), np.uint8)
        dilated_mask = cv2.dilate(person_mask, kernel, iterations=1)
        blur_border = cv2.GaussianBlur(dilated_mask, (21, 21), 0)

        # Chuyển đổi mask thành 3 kênh
        person_mask_3ch = cv2.cvtColor(person_mask, cv2.COLOR_GRAY2BGR)
        blur_border_3ch = cv2.cvtColor(blur_border, cv2.COLOR_GRAY2BGR) / 255.0

        # Kết hợp hình nền mờ và đối tượng chính sắc nét
        main_highlight = blurred_bg.copy()
        main_highlight = np.where(person_mask_3ch > 0, img, main_highlight)

        # Thêm hiệu ứng glow xung quanh đối tượng chính
        glow_effect = np.where(blur_border_3ch > 0,
                               img * blur_border_3ch + blurred_bg * (1 - blur_border_3ch),
                               blurred_bg)

        # Đánh dấu box và thêm nhãn
        color = (0, 255, 255)  # Yellow for main person
        #cv2.rectangle(main_highlight, (x1, y1), (x2, y2), color, 3)
        #cv2.putText(main_highlight, "MAIN SUBJECT", (x1, y1 - 10),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        main_highlight = img.copy()

    # Create composition analysis visualization
    comp_viz = img.copy()

    # Draw rule of thirds grid
    for i in range(1, 3):
        cv2.line(comp_viz, (0, int(height * i / 3)), (width, int(height * i / 3)), (255, 255, 255), 1)
        cv2.line(comp_viz, (int(width * i / 3), 0), (int(width * i / 3), height), (255, 255, 255), 1)

    # Draw key subjects with prominence
    if "key_subjects" in sports_analysis:
        for idx, subject in enumerate(sports_analysis['key_subjects']):
            box = subject['box']
            x1, y1, x2, y2 = box

            # Xác định nếu là đối tượng chính (người)
            is_main_subject = (idx == main_person_idx)
            subject_id = idx + 1

            if is_main_subject:
                # Highlight đối tượng chính với màu nổi bật
                color = (0, 255, 255)  # Yellow
                thickness = 3
            else:
                # Color based on prominence - more red = more important
                prominence = min(1.0, subject['prominence'] * 10)  # Scale for visibility
                color = (0, int(255 * (1 - prominence)), int(255 * prominence))
                thickness = 2

            cv2.rectangle(comp_viz, (x1, y1), (x2, y2), color, thickness)

            # Hiển thị ID, điểm sắc nét và chỉ số prominence
            label_text = f"ID:{subject_id} P:{subject['prominence']:.2f} S:{subject.get('sharpness', 0):.2f}"
            if is_main_subject:
                label_text = f"ID:{subject_id} MAIN P:{subject['prominence']:.2f} S:{subject.get('sharpness', 0):.2f}"

            cv2.putText(comp_viz, label_text,
                        (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Tạo heatmap độ sắc nét cải tiến
    try:
        # Tạo sharpness heatmap cho toàn bộ ảnh
        sharpness_overlay, sharpness_heatmap_raw = create_sharpness_heatmap(img)
        sharpness_viz = sharpness_overlay.copy()

        # Sử dụng colormap
        from matplotlib import cm
        jet_colormap = cm.get_cmap('jet')

        # Vẽ bounding boxes với sharpness scores
        for i, box in enumerate(detections['boxes']):
            if i < len(sharpness_scores):
                x1, y1, x2, y2 = box
                sharpness = sharpness_scores[i]

                # Màu dựa trên độ sắc nét
                color_rgba = jet_colormap(sharpness)
                color_rgb = tuple(int(255 * c) for c in color_rgba[:3])
                color = (color_rgb[2], color_rgb[1], color_rgb[0])  # BGR for OpenCV

                # Vẽ border dày hơn cho objects có sharpness cao
                border_thickness = max(2, int(sharpness * 5))
                cv2.rectangle(sharpness_viz, (x1, y1), (x2, y2), color, border_thickness)

                # Thêm nhãn với background
                label_text = f"Sharp: {sharpness:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                # Background cho text
                cv2.rectangle(sharpness_viz, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                cv2.putText(sharpness_viz, label_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        print("Sharpness heatmap created successfully")

    except Exception as e:
        print(f"Error creating sharpness heatmap: {e}")
        # Fallback to original method
        try:
            # Tạo sharpness heatmap cho toàn bộ ảnh
            sharpness_overlay, sharpness_heatmap_raw = create_sharpness_heatmap(img)
            sharpness_viz = sharpness_overlay.copy()

            # Sử dụng colormap
            from matplotlib import cm
            jet_colormap = cm.get_cmap('jet')

            # Vẽ bounding boxes với sharpness scores
            for i, box in enumerate(detections['boxes']):
                if i < len(sharpness_scores):
                    x1, y1, x2, y2 = box
                    sharpness = sharpness_scores[i]

                    # Màu dựa trên độ sắc nét
                    color_rgba = jet_colormap(sharpness)
                    color_rgb = tuple(int(255 * c) for c in color_rgba[:3])
                    color = (color_rgb[2], color_rgb[1], color_rgb[0])  # BGR for OpenCV

                    # Vẽ border dày hơn cho objects có sharpness cao
                    border_thickness = max(2, int(sharpness * 5))
                    cv2.rectangle(sharpness_viz, (x1, y1), (x2, y2), color, border_thickness)

                    # Thêm nhãn với background
                    label_text = f"Sharp: {sharpness:.2f}"
                    (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                    # Background cho text
                    cv2.rectangle(sharpness_viz, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                    cv2.putText(sharpness_viz, label_text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            print("Sharpness heatmap created successfully")

        except Exception as e:
            print(f"Error creating sharpness heatmap: {e}")
            # Fallback to original method
            sharpness_viz = img.copy()

            from matplotlib import cm
            jet_colormap = cm.get_cmap('jet')

            for i, box in enumerate(detections['boxes']):
                if i < len(sharpness_scores):
                    x1, y1, x2, y2 = box
                    sharpness = sharpness_scores[i]

                    color_rgba = jet_colormap(sharpness)
                    color_rgb = tuple(int(255 * c) for c in color_rgba[:3])
                    color = (color_rgb[2], color_rgb[1], color_rgb[0])

                    overlay = sharpness_viz.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    alpha = 0.4 + 0.3 * sharpness
                    cv2.addWeighted(overlay, alpha, sharpness_viz, 1 - alpha, 0, sharpness_viz)

                    cv2.rectangle(sharpness_viz, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(sharpness_viz, f"Sharp: {sharpness:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        from matplotlib import cm
        jet_colormap = cm.get_cmap('jet')

        for i, box in enumerate(detections['boxes']):
            if i < len(sharpness_scores):
                x1, y1, x2, y2 = box
                sharpness = sharpness_scores[i]

                color_rgba = jet_colormap(sharpness)
                color_rgb = tuple(int(255 * c) for c in color_rgba[:3])
                color = (color_rgb[2], color_rgb[1], color_rgb[0])

                overlay = sharpness_viz.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                alpha = 0.4 + 0.3 * sharpness
                cv2.addWeighted(overlay, alpha, sharpness_viz, 1 - alpha, 0, sharpness_viz)

                cv2.rectangle(sharpness_viz, (x1, y1), (x2, y2), color, 3)
                cv2.putText(sharpness_viz, f"Sharp: {sharpness:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # THÊM MỚI: Hiển thị keypoints từ pose estimation nếu có
    pose_viz = img.copy()

    # Kiểm tra xác thực cấu trúc dữ liệu pose_analysis
    # Kiểm tra xác thực cấu trúc dữ liệu pose_analysis
    if 'pose_analysis' in sports_analysis and isinstance(sports_analysis['pose_analysis'], dict) and 'poses' in \
            sports_analysis['pose_analysis']:
        poses = sports_analysis['pose_analysis']['poses']

        # CHỈ lấy một pose duy nhất - pose của main subject
        main_subject_pose = None

        # Nếu có main subject mask thì dùng mask để tìm pose
        main_subject_pose = None

        # Kiểm tra xem có main subject mask không
        if 'main_subject_mask' in sports_analysis and sports_analysis['main_subject_mask'] is not None:
            main_mask = sports_analysis['main_subject_mask']

            # Đảm bảo mask có kích thước phù hợp với ảnh
            if main_mask.shape[:2] != (height, width):
                # Resize mask nếu cần
                main_mask = cv2.resize(main_mask, (width, height))

            best_pose = None
            max_in_mask = 0
            max_ratio = 0

            print(f"Đang kiểm tra {len(poses)} poses với mask")

            # Duyệt qua từng pose và kiểm tra keypoints trong mask
            for idx, pose in enumerate(poses):
                if 'keypoints' in pose and pose['keypoints']:
                    # Đếm số keypoint nằm trong mask
                    count_in_mask = 0
                    total_keypoints = len(pose['keypoints'])

                    for kp in pose['keypoints']:
                        if kp['confidence'] < 0.2:  # Bỏ qua keypoints có độ tin cậy thấp
                            continue

                        x, y = int(kp['x']), int(kp['y'])

                        # Kiểm tra x,y có nằm trong phạm vi mask không
                        if 0 <= x < width and 0 <= y < height:
                            # Kiểm tra keypoint có nằm trong mask không
                            if main_mask[y, x] > 0.5:
                                count_in_mask += 1

                    # Tính tỷ lệ keypoints trong mask
                    ratio = count_in_mask / total_keypoints if total_keypoints > 0 else 0

                    print(f"Pose {idx}: {count_in_mask}/{total_keypoints} keypoints trong mask ({ratio * 100:.1f}%)")

                    # Chỉ chọn pose có ít nhất 30% keypoints trong mask
                    if ratio > max_ratio and ratio > 0.3:
                        max_ratio = ratio
                        max_in_mask = count_in_mask
                        best_pose = pose
                        print(f"  -> Pose {idx} hiện là pose tốt nhất với {ratio * 100:.1f}% keypoints trong mask")

            if best_pose:
                print(f"Đã tìm thấy pose phù hợp với mask: {max_in_mask} keypoints, {max_ratio * 100:.1f}%")
                main_subject_pose = best_pose
            else:
                print(f"Không tìm thấy pose nào có đủ keypoints trong mask (ngưỡng 30%)")

        # Backup: Nếu không tìm được pose dựa trên mask, thử dùng IoU với box
        if main_subject_pose is None and main_person is not None:
            print("Thử tìm pose dựa trên IoU với bounding box")
            main_x1, main_y1, main_x2, main_y2 = main_person['box']
            best_iou = 0

            for pose in poses:
                if 'bbox' in pose and pose['bbox']:
                    p_x1, p_y1, p_x2, p_y2 = pose['bbox']

                    # Tính IoU (Intersection over Union)
                    x_left = max(main_x1, p_x1)
                    y_top = max(main_y1, p_y1)
                    x_right = min(main_x2, p_x2)
                    y_bottom = min(main_y2, p_y2)

                    if x_right <= x_left or y_bottom <= y_top:
                        continue

                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    main_area = (main_x2 - main_x1) * (main_y2 - main_y1)
                    pose_area = (p_x2 - p_x1) * (p_y2 - p_y1)
                    union = main_area + pose_area - intersection

                    iou = intersection / union if union > 0 else 0
                    print(f"IoU với pose bbox: {iou:.2f}")

                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        main_subject_pose = pose

        # Backup: nếu không tìm được bằng mask, dùng bounding box
        if main_subject_pose is None and main_person is not None:
            main_x1, main_y1, main_x2, main_y2 = main_person['box']
            (main_x1 + main_x2) / 2
            (main_y1 + main_y2) / 2

            # Tìm pose có bbox trùng nhiều nhất với main person box
            best_iou = 0
            for pose in poses:
                if 'bbox' in pose and pose['bbox']:
                    p_x1, p_y1, p_x2, p_y2 = pose['bbox']

                    # Tính IoU
                    x_left = max(main_x1, p_x1)
                    y_top = max(main_y1, p_y1)
                    x_right = min(main_x2, p_x2)
                    y_bottom = min(main_y2, p_y2)

                    if x_right < x_left or y_bottom < y_top:
                        continue

                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    main_area = (main_x2 - main_x1) * (main_y2 - main_y1)
                    pose_area = (p_x2 - p_x1) * (p_y2 - p_y1)
                    union = main_area + pose_area - intersection

                    iou = intersection / union if union > 0 else 0

                    if iou > best_iou:
                        best_iou = iou
                        main_subject_pose = pose

        # Nếu tìm được pose của main subject, chỉ xử lý pose đó
        if main_subject_pose:
            poses = [main_subject_pose]  # Chỉ xử lý pose của main subject
        else:
            poses = []  # Không có pose nào khớp với main subject

        # Nếu không tìm thấy theo IoU, lấy người có bbox lớn nhất/ở giữa nhất
        if main_subject_pose is None and poses:
            largest_area = 0
            center_pose = None

            for pose in poses:
                if 'bbox' in pose and pose['bbox']:
                    p_x1, p_y1, p_x2, p_y2 = pose['bbox']
                    area = (p_x2 - p_x1) * (p_y2 - p_y1)

                    # Ưu tiên người ở giữa
                    p_center_x = (p_x1 + p_x2) / 2
                    p_center_y = (p_y1 + p_y2) / 2
                    center_score = 1 - (abs(p_center_x - width / 2) / width + abs(p_center_y - height / 2) / height) / 2

                    # Kết hợp diện tích và vị trí
                    score = area * center_score

                    if score > largest_area:
                        largest_area = score
                        center_pose = pose

            main_subject_pose = center_pose

        # Xử lý chỉ với main_subject_pose
        if main_subject_pose:
            poses = [main_subject_pose]  # Chỉ xử lý pose của main subject

        # Định nghĩa các cặp keypoint để vẽ khung xương
        skeleton = [
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            (5, 6),  # Shoulders
            (5, 11), (6, 12),  # Body
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16),  # Right leg
            (11, 12)  # Hips
        ]

        print(f"Số người được phát hiện pose: {len(poses)}")

        for person in poses:
            # Kiểm tra và debug thông tin keypoints
            print(f"Số keypoints của người: {len(person['keypoints'])}")

            # Tạo dict để lưu các keypoint theo id
            keypoints = {kp['id']: (int(kp['x']), int(kp['y'])) for kp in person['keypoints']}

            # Vẽ các keypoint
            for kp in person['keypoints']:
                if kp['confidence'] < 0.2:  # Bỏ qua các điểm có độ tin cậy quá thấp
                    continue

                x, y = int(kp['x']), int(kp['y'])

                # Màu sắc cho các keypoint khác nhau
                if kp['id'] <= 4:  # Vùng đầu
                    color = (255, 0, 0)  # Xanh dương
                elif 5 <= kp['id'] <= 10:  # Vùng tay
                    color = (0, 255, 255)  # Vàng
                else:  # Vùng chân
                    color = (0, 255, 0)  # Xanh lá

                # Vẽ điểm lớn hơn, với viền đen để dễ nhìn hơn
                cv2.circle(pose_viz, (x, y), 7, (0, 0, 0), -1)  # Viền đen
                cv2.circle(pose_viz, (x, y), 5, color, -1)  # Điểm màu

                # TẮT hiển thị tên keypoint để tránh rối mắt
                # Chỉ hiển thị confidence bên cạnh điểm
                # if kp['confidence'] > 0.6:  # Chỉ hiển thị cho điểm có độ tin cậy cao
                #     cv2.putText(pose_viz, f"{kp['confidence']:.2f}", (x+5, y),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Vẽ skeleton
            for kp1_id, kp2_id in skeleton:
                if kp1_id in keypoints and kp2_id in keypoints:
                    pt1 = keypoints[kp1_id]
                    pt2 = keypoints[kp2_id]

                    # Kiểm tra khoảng cách để tránh vẽ các đường quá dài
                    distance = np.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))
                    # Khoảng cách tối đa hợp lý giữa các keypoint (có thể điều chỉnh)
                    max_distance = width * 0.5  # 50% chiều rộng ảnh

                    if distance > max_distance:
                        continue  # Bỏ qua các skeleton quá dài

                    # Sử dụng màu khác nhau cho các phần khác nhau của cơ thể
                    if (kp1_id <= 4 and kp2_id <= 4):  # Phần đầu
                        line_color = (255, 0, 0)
                    elif (5 <= kp1_id <= 10) or (5 <= kp2_id <= 10):  # Phần tay
                        line_color = (0, 255, 255)
                    else:  # Phần chân
                        line_color = (0, 255, 0)

                    # Vẽ đường với độ dày lớn hơn
                    cv2.line(pose_viz, pt1, pt2, (0, 0, 0), 5)  # Đường viền đen
                    cv2.line(pose_viz, pt1, pt2, line_color, 3)  # Đường màu
    else:
        print("Không tìm thấy dữ liệu pose_analysis hoặc cấu trúc dữ liệu không đúng")
        print(
            f"sports_analysis keys: {sports_analysis.keys() if isinstance(sports_analysis, dict) else type(sports_analysis)}")
        if isinstance(sports_analysis, dict) and 'pose_analysis' in sports_analysis:
            print(f"pose_analysis keys: {sports_analysis['pose_analysis'].keys()}")

    # Hiển thị biểu cảm khuôn mặt
    face_emotion_viz = None
    if facial_analysis and facial_analysis.get('has_faces', False):
        try:
            # Tìm ảnh khuôn mặt
            face_img = None
            if 'face_path' in facial_analysis:
                face_path = facial_analysis['face_path']
                print(f"Đọc ảnh khuôn mặt từ: {face_path}")
                if os.path.exists(face_path):
                    face_img = cv2.imread(face_path)
                    if face_img is not None:
                        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        print(f"Đã đọc ảnh khuôn mặt thành công, kích thước: {face_img.shape}")
                    else:
                        print(f"Không thể đọc ảnh khuôn mặt từ {face_path}")

            # Nếu không có face_path hoặc không đọc được, thử tìm trực tiếp trong thư mục debug
            if face_img is None:
                fallback_path = "face_debug/best_face.jpg"
                if os.path.exists(fallback_path):
                    face_img = cv2.imread(fallback_path)
                    if face_img is not None:
                        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        print(f"Đã đọc ảnh khuôn mặt từ đường dẫn dự phòng: {fallback_path}")
                    else:
                        print(f"Không thể đọc ảnh khuôn mặt từ đường dẫn dự phòng")
                else:
                    print(f"Không tìm thấy file ảnh khuôn mặt dự phòng: {fallback_path}")

            # Nếu đọc được ảnh khuôn mặt, tạo hiển thị biểu cảm
            if face_img is not None:
                # Đảm bảo kích thước hợp lý cho ảnh khuôn mặt
                h, w = face_img.shape[:2]
                display_width = 300
                display_height = int((h / w) * display_width) if w > 0 else 300

                # Điều chỉnh kích thước ảnh khuôn mặt cho phù hợp
                face_display = cv2.resize(face_img, (display_width, display_height),
                                          interpolation=cv2.INTER_AREA)

                # Lấy thông tin cảm xúc
                emotion = facial_analysis.get('dominant_emotion', 'unknown')
                intensity = facial_analysis.get('emotion_intensity', 0)

                # Tạo hình ảnh hiển thị biểu cảm
                face_emotion_viz = np.ones((400, 400, 3), dtype=np.uint8) * 255  # Tạo nền trắng

                # Hiển thị ảnh khuôn mặt ở giữa
                y_offset = 50
                x_offset = (400 - display_width) // 2
                face_emotion_viz[y_offset:y_offset + display_height, x_offset:x_offset + display_width] = face_display

                # Hiển thị thông tin cảm xúc
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(face_emotion_viz, f"Emotion: {emotion.upper()}", (20, 30),
                            font, 0.7, (0, 0, 0), 2)
                cv2.putText(face_emotion_viz, f"Intensity: {intensity:.2f}", (20, y_offset + display_height + 30),
                            font, 0.7, (0, 0, 0), 2)

                print(f"Đã tạo hiển thị biểu cảm khuôn mặt thành công")
            else:
                print("Không thể đọc ảnh khuôn mặt từ bất kỳ nguồn nào")
        except Exception as e:
            import traceback
            print(f"Lỗi khi tạo hiển thị biểu cảm: {str(e)}")
            print(traceback.format_exc())

    # Lưu các thành phần riêng biệt
    # Depth map
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_map, cmap='plasma')
    plt.title("Depth Map")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("depth_map.png", dpi=150)
    plt.close()

    # Detections
    plt.figure(figsize=(8, 6))
    plt.imshow(det_viz)
    plt.title(f"Detections ({detections['athletes']} athletes)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("detections.png", dpi=150)
    plt.close()

    # Main subject highlight
    plt.figure(figsize=(8, 6))
    plt.imshow(main_highlight)
    plt.title("Main Subject Highlight")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("main_subject_highlight.png", dpi=150)
    plt.close()

    # Sharpness heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(sharpness_viz)
    plt.title("Sharpness Heatmap")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("sharpness_heatmap.png", dpi=150)
    plt.close()

    # Composition analysis
    plt.figure(figsize=(8, 6))
    plt.imshow(comp_viz)
    plt.title("Composition Analysis")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("composition_analysis.png", dpi=150)
    plt.close()

    # THÊM MỚI: Lưu Pose Estimation
    plt.figure(figsize=(8, 6))
    plt.imshow(pose_viz)
    plt.title("Pose Estimation")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("pose_estimation.png", dpi=150)
    plt.close()

    # Hiển thị với bố cục nâng cao
    fig = plt.figure(figsize=(18, 12))

    # THAY ĐỔI: Cập nhật GridSpec để thêm pose estimation
    grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)

    # Ảnh gốc
    ax_orig = fig.add_subplot(grid[0, 0])
    ax_orig.imshow(img)
    ax_orig.set_title("Original Image")
    ax_orig.axis('off')

    # Main subject highlight (lớn nhất - ở giữa)
    ax_main = fig.add_subplot(grid[0:2, 1:3])
    ax_main.imshow(main_highlight)
    ax_main.set_title("Main Subject Highlight")
    ax_main.axis('off')

    # Detections
    ax_det = fig.add_subplot(grid[0, 3])
    ax_det.imshow(det_viz)
    ax_det.set_title(f"Detections ({detections['athletes']} athletes)")
    ax_det.axis('off')

    # Depth map
    ax_depth = fig.add_subplot(grid[1, 0])
    ax_depth.imshow(depth_map, cmap='plasma')
    ax_depth.set_title("Depth Map")
    ax_depth.axis('off')

    # Composition analysis
    ax_comp = fig.add_subplot(grid[1, 3])
    ax_comp.imshow(comp_viz)
    ax_comp.set_title("Composition Analysis")
    ax_comp.axis('off')

    # Sharpness heatmap
    ax_sharp = fig.add_subplot(grid[2, 0:2])
    ax_sharp.imshow(sharpness_viz)
    ax_sharp.set_title("Sharpness Heatmap")
    ax_sharp.axis('off')

    # THÊM MỚI: Pose estimation visualization
    ax_pose = fig.add_subplot(grid[2:4, 2:4])  # ĐIỀU CHỈNH VỊ TRÍ HIỂN THỊ POSE
    ax_pose.imshow(pose_viz)
    ax_pose.set_title("Pose Estimation")
    ax_pose.axis('off')

    # Face analysis nâng cao
    if face_emotion_viz is not None:
        ax_face = fig.add_subplot(grid[3, 0:2])  # ĐIỀU CHỈNH VỊ TRÍ HIỂN THỊ FACE
        ax_face.imshow(face_emotion_viz)

        # Hiển thị tiêu đề chi tiết hơn
        emotion = facial_analysis.get('dominant_emotion', 'unknown')
        intensity = facial_analysis.get('emotion_intensity', 0)
        original_emotion = facial_analysis.get('original_emotion', emotion)

        if original_emotion != emotion:
            title = f"Face: {emotion.upper()} ({intensity:.2f}) [orig: {original_emotion}]"
        else:
            title = f"Face: {emotion.upper()} ({intensity:.2f})"

        ax_face.set_title(title)
        ax_face.axis('off')
    else:
        # Hiển thị thông báo rõ ràng khi không phát hiện được khuôn mặt
        ax_info = fig.add_subplot(grid[3, 0:2])  # ĐIỀU CHỈNH VỊ TRÍ HIỂN THỊ LỖI FACE

        # Tạo thông báo "No face detected"
        if not facial_analysis or not facial_analysis.get('has_faces', False):
            ax_info.text(0.5, 0.5, "NO FACE DETECTED\nFacial analysis skipped",
                         fontsize=16, color='red', fontweight='bold',
                         horizontalalignment='center', verticalalignment='center')
        else:
            # Trường hợp có facial_analysis nhưng không có ảnh
            emotion = facial_analysis.get('dominant_emotion', 'unknown')
            intensity = facial_analysis.get('emotion_intensity', 0)
            text = f"Face Emotion: {emotion}\nIntensity: {intensity:.2f}"
            if 'original_emotion' in facial_analysis and facial_analysis['original_emotion'] != emotion:
                text += f"\nOriginal: {facial_analysis['original_emotion']}"

            ax_info.text(0.5, 0.5, text, horizontalalignment='center',
                         verticalalignment='center', fontsize=14)

        ax_info.axis('off')
        ax_info.set_title("Facial Expression")

    plt.savefig("sports_analysis_results.png", dpi=150)
    plt.close()

    # Print detailed analysis
    print("\n==== SPORTS IMAGE ANALYSIS ====")
    print(
        f"Detected {detections['athletes']} athletes and {len(detections['classes']) - detections['athletes']} other objects")

    if "sport_type" in composition_analysis:
        print(f"\nSport type: {composition_analysis['sport_type']}")

        # Thêm thông tin về môi trường nếu có
        if 'environment_analysis' in composition_analysis:
            env = composition_analysis['environment_analysis']
            if env['detected_environments']:
                print(f"Detected environment: {', '.join(env['detected_environments'])}")
            if env['surface_type'] != 'unknown':
                print(f"Surface type: {env['surface_type']}")
            if env['dominant_colors']:
                colors = [f"{color} ({ratio:.2f})" for color, ratio in env['dominant_colors']]
                print(f"Dominant colors: {', '.join(colors)}")

    if detections['athletes'] > 0:
        print("\nPlayer Analysis:")
        print(f"- Number of players: {detections['athletes']}")
        if detections['athletes'] > 1:
            print(f"- Player dispersion: {sports_analysis['player_dispersion']:.2f}")

    print("\nAction Analysis:")
    print(
        f"- Equipment detected: {', '.join(action_analysis['equipment_types']) if action_analysis['equipment_types'] else 'None'}")
    print(f"- Action level: {action_analysis['action_quality']} ({action_analysis['action_level']:.2f})")

    print("\nComposition Analysis:")
    print(f"- Framing quality: {composition_analysis['framing_quality']}")

    # Hiển thị chi tiết framing analysis nếu có
    if 'framing_analysis' in composition_analysis and 'overall_score' in composition_analysis['framing_analysis']:
        framing = composition_analysis['framing_analysis']
        print(f"  * Overall score: {framing['overall_score']:.3f}")
        print(f"  * Position score: {framing['position_score']:.3f}")
        print(f"  * Size score: {framing['size_score']:.3f}")
        print(f"  * Breathing room score: {framing['breathing_score']:.3f}")
        print(f"  * Subject size ratio: {framing['size_ratio']:.3f}")
        print(f"  * Min margin: {framing['min_margin']:.3f}")

    if composition_analysis['recommended_crop']:
        crop = composition_analysis['recommended_crop']
        direction_x = "right" if crop['shift_x'] < 0 else "left"
        direction_y = "down" if crop['shift_y'] < 0 else "up"
        print(
            f"- Recommended crop: Shift {abs(crop['shift_x']) * 100:.1f}% {direction_x} and {abs(crop['shift_y']) * 100:.1f}% {direction_y}")

    # Key subjects with sharpness
    if sports_analysis['key_subjects']:
        print("\nKey Subjects by Prominence:")
        for i, subject in enumerate(sports_analysis['key_subjects']):
            main_tag = " (MAIN SUBJECT)" if main_person_idx == i else ""
            print(
                f"{i + 1}. {subject['class']}{main_tag} (Prominence: {subject['prominence']:.2f}, Sharpness: {subject.get('sharpness', 0):.2f})")

            # Chi tiết về độ sắc nét nếu có
            if 'sharpness_details' in subject and subject['sharpness_details']:
                details = subject['sharpness_details']
                print(f"   - Laplacian Variance: {details['laplacian_var']:.2f}")
                print(f"   - Sobel Mean: {details['sobel_mean']:.2f}")

    # HIỂN THỊ PHÂN TÍCH BIỂU CẢM CẢI TIẾN
    if facial_analysis and facial_analysis.get('has_faces', False):
        print("\nFacial Expression Analysis (Advanced):")
        print(f"- Dominant emotion: {facial_analysis['dominant_emotion']}")
        if 'original_emotion' in facial_analysis and facial_analysis['original_emotion'] != facial_analysis[
            'dominant_emotion']:
            print(f"- Original detected emotion: {facial_analysis['original_emotion']}")
        print(f"- Emotion intensity: {facial_analysis['emotion_intensity']:.2f}")
        print(f"- Emotional value: {facial_analysis['emotional_value']}")
        print(f"- Sport context: {facial_analysis.get('sport_context', 'Unknown')}")

        # Hiển thị chi tiết điểm số cảm xúc nếu có
        if 'contextual_scores' in facial_analysis:
            print("\nDetailed Emotion Scores:")
            for emotion, score in facial_analysis['contextual_scores'].items():
                print(f"  - {emotion}: {score:.3f}")
    else:
        print("\nFacial Expression Analysis: No valid face detected")

    # Save analysis results
    with open("analysis_results.txt", "w") as f:
        f.write("\n==== SPORTS IMAGE ANALYSIS ====\n")
        f.write(
            f"Detected {detections['athletes']} athletes and {len(detections['classes']) - detections['athletes']} other objects\n")

        if "sport_type" in composition_analysis:
            f.write(f"\nSport type: {composition_analysis['sport_type']}\n")

        if detections['athletes'] > 0:
            f.write("\nPlayer Analysis:\n")
            f.write(f"- Number of players: {detections['athletes']}\n")
            if detections['athletes'] > 1:
                f.write(f"- Player dispersion: {sports_analysis['player_dispersion']:.2f}\n")

        f.write("\nAction Analysis:\n")
        f.write(
            f"- Equipment detected: {', '.join(action_analysis['equipment_types']) if action_analysis['equipment_types'] else 'None'}\n")
        f.write(f"- Action quality: {action_analysis['action_quality']} ({action_analysis['action_level']:.2f})\n")

        # Thêm chi tiết framing analysis
        if 'framing_analysis' in composition_analysis and 'overall_score' in composition_analysis['framing_analysis']:
            framing = composition_analysis['framing_analysis']
            f.write(f"  * Overall score: {framing['overall_score']:.3f}\n")
            f.write(f"  * Position score: {framing['position_score']:.3f}\n")
            f.write(f"  * Size score: {framing['size_score']:.3f}\n")
            f.write(f"  * Breathing room score: {framing['breathing_score']:.3f}\n")
            f.write(f"  * Subject size ratio: {framing['size_ratio']:.3f}\n")
            f.write(f"  * Min margin: {framing['min_margin']:.3f}\n")

        if composition_analysis['recommended_crop']:
            crop = composition_analysis['recommended_crop']
            direction_x = "right" if crop['shift_x'] < 0 else "left"
            direction_y = "down" if crop['shift_y'] < 0 else "up"
            f.write(
                f"- Recommended crop: Shift {abs(crop['shift_x']) * 100:.1f}% {direction_x} and {abs(crop['shift_y']) * 100:.1f}% {direction_y}\n")

        # Thêm thông tin độ sắc nét khi lưu kết quả
        if sports_analysis['key_subjects']:
            f.write("\nKey Subjects by Prominence:\n")
            for i, subject in enumerate(sports_analysis['key_subjects']):
                main_tag = " (MAIN SUBJECT)" if main_person_idx == i else ""
                f.write(
                    f"{i + 1}. {subject['class']}{main_tag} (Prominence: {subject['prominence']:.2f}, Sharpness: {subject.get('sharpness', 0):.2f})\n")

        # Lưu phân tích biểu cảm nâng cao
        if facial_analysis and facial_analysis.get('has_faces', False):
            f.write("\nFacial Expression Analysis (Advanced):\n")
            f.write(f"- Dominant emotion: {facial_analysis['dominant_emotion']}\n")
            if 'original_emotion' in facial_analysis and facial_analysis['original_emotion'] != facial_analysis[
                'dominant_emotion']:
                f.write(f"- Original detected emotion: {facial_analysis['original_emotion']}\n")
            f.write(f"- Emotion intensity: {facial_analysis['emotion_intensity']:.2f}\n")
            f.write(f"- Emotional value: {facial_analysis['emotional_value']}\n")
            f.write(f"- Sport context: {facial_analysis.get('sport_context', 'Unknown')}\n")

            # Chi tiết điểm số cảm xúc
            if 'contextual_scores' in facial_analysis:
                f.write("\nDetailed Emotion Scores:\n")
                for emotion, score in facial_analysis['contextual_scores'].items():
                    f.write(f"  - {emotion}: {score:.3f}\n")
        else:
            f.write("\nFacial Expression Analysis: No valid face detected\n")

        # Lưu caption
        if caption:
            f.write("\nImage Caption:\n")
            f.write(caption + "\n")

def analyze_sports_image(file_path):
    """
    Main function to analyze sports images with enhanced pose detection
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Dictionary containing complete analysis results
    """
    t_start = time.time()

    # Load models
    print("Loading models...")
    midas, yolo, yolo_seg, device = load_models()
    print("Models loaded successfully.")

    img_data = preprocess_image(file_path)
    print(f"Image preprocessed: {file_path}")

    # Step 1: Object detection with YOLO
    print("Detecting objects...")
    detections = detect_sports_objects(yolo, img_data)
    print(f"Found {len(detections['classes'])} objects, including {detections['athletes']} athletes")

    # Step 2: Generate depth map
    print("Generating depth map...")
    depth_map, depth_mask, depth_contour = generate_depth_map(midas, img_data)
    print("Depth map generated.")

    # Step 3: Analyze sports scene with sharpness detection
    print("Analyzing sports scene with sharpness detection...")
    sports_analysis = analyze_sports_scene(detections, depth_map, img_data, yolo_seg)

    # Step 4: Analyze action quality
    print("Analyzing action quality...")
    action_analysis = analyze_action_quality(detections, img_data)

    # Step 5: Sports composition analysis
    print("Analyzing composition...")
    composition_analysis = analyze_sports_composition(detections, {
        'sports_analysis': sports_analysis,
        'depth_map': depth_map
    }, img_data)

    # Step 6: Enhanced facial expression analysis
    print("Starting advanced facial expression analysis...")
    try:
        facial_analysis = analyze_facial_expression_advanced(
            detections,
            img_data,
            depth_map=depth_map,
            sports_analysis={
                'sports_analysis': sports_analysis,
                'composition_analysis': composition_analysis
            }
        )
        print("Advanced facial expression analysis successful:", facial_analysis.get('has_faces', False))
    except Exception as e:
        print(f"Error in facial expression analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        facial_analysis = {'has_faces': False, 'error': str(e)}

    img_data['detections'] = detections
    # Step 6.5: Enhanced pose detection with sport context
    print("Detecting and analyzing human poses...")
    pose_results = detect_human_pose(
        img_data,
        conf_threshold=0.15,
        main_subject_box=sports_analysis.get('key_subjects', [{}])[0].get('box'),
        sport_type=composition_analysis.get('sport_type')
    )

    
    # Update sports_analysis with pose results
    sports_analysis['pose_analysis'] = pose_results

    # Create final analysis result
    analysis_result = {
        'detections': detections,
        'sports_analysis': sports_analysis,
        'action_analysis': action_analysis,
        'composition_analysis': composition_analysis,
        'facial_analysis': facial_analysis,
        'pose_analysis': pose_results
    }

    # Generate visualization
    print("Visualizing results...")
    visualize_sports_results(
        img_data,
        detections,
        depth_map,
        sports_analysis,
        action_analysis,
        composition_analysis,
        facial_analysis
    )

    t_end = time.time()
    print(f"\nAnalysis completed in {t_end - t_start:.2f} seconds")

    # Generate caption
    caption = generate_sports_caption(analysis_result)
    print(f"\nCaption: {caption}")

    # Add caption to results
    analysis_result['caption'] = caption

    return analysis_result

def analyze_movement_dynamics(keypoints, prev_keypoints=None, sport_type=None):
    """
    Analyze movement dynamics from pose keypoints
    
    Args:
        keypoints: Current frame keypoints
        prev_keypoints: Previous frame keypoints (if available)
        sport_type: Type of sport being analyzed
    
    Returns:
        dict: Detailed movement analysis
    """
    dynamics = {
        'speed': None,
        'direction': None,
        'force': None,
        'rotation': None,
        'extension': None,
        'balance': None,
        'specific_action': None,
        'height': None,
        'stance': None
    }
    
    def calculate_limb_velocity(current_points, previous_points, joint_pairs):
        """Calculate velocity between joint pairs"""
        if not (current_points and previous_points):
            return 0
        velocities = []
        for start, end in joint_pairs:
            start_key = f"{start}_shoulder" if "shoulder" in start else f"{start}_knee" if "knee" in start else f"{start}_ankle"
            end_key = f"{end}_shoulder" if "shoulder" in end else f"{end}_knee" if "knee" in end else f"{end}_ankle"
            
            if all(p in current_points and p in previous_points 
                  for p in [start_key, end_key]):
                curr_vec = (
                    float(current_points[end_key]['x']) - float(current_points[start_key]['x']),
                    float(current_points[end_key]['y']) - float(current_points[start_key]['y'])
                )
                prev_vec = (
                    float(previous_points[end_key]['x']) - float(previous_points[start_key]['x']),
                    float(previous_points[end_key]['y']) - float(previous_points[start_key]['y'])
                )
                velocity = math.sqrt(
                    (curr_vec[0] - prev_vec[0])**2 + 
                    (curr_vec[1] - prev_vec[1])**2
                )
                velocities.append(velocity)
        return max(velocities) if velocities else 0

    try:
        # Handle nested keypoints structure
        if isinstance(keypoints, dict) and 'poses' in keypoints:
            if keypoints['poses']:
                keypoints = keypoints['poses'][0].get('keypoints', [])
        
        # Convert keypoints to dictionary for easier access
        current_points = {}
        for kp in keypoints:
            if isinstance(kp, dict) and 'name' in kp:
                current_points[kp['name']] = {
                    'x': float(kp['x']),
                    'y': float(kp['y']),
                    'confidence': float(kp.get('confidence', 0))
                }
        
        # Convert previous keypoints if available
        previous_points = {}
        if prev_keypoints:
            if isinstance(prev_keypoints, dict) and 'poses' in prev_keypoints:
                if prev_keypoints['poses']:
                    prev_keypoints = prev_keypoints['poses'][0].get('keypoints', [])
            
            for kp in prev_keypoints:
                if isinstance(kp, dict) and 'name' in kp:
                    previous_points[kp['name']] = {
                        'x': float(kp['x']),
                        'y': float(kp['y']),
                        'confidence': float(kp.get('confidence', 0))
                    }
        
        # Analyze based on sport type
        if sport_type:
            sport_type = sport_type.lower()
            
            if sport_type in ['tennis', 'badminton', 'baseball']:
                # Analyze swing dynamics
                arm_joints = [
                    ('left', 'right'),  # For shoulders
                    ('right', 'left')   # For arms
                ]
                swing_velocity = calculate_limb_velocity(
                    current_points, previous_points, arm_joints
                )
                
                # Determine swing direction using shoulders and wrists
                if all(k in current_points for k in ['left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist']):
                    shoulder_mid_x = (current_points['left_shoulder']['x'] + current_points['right_shoulder']['x']) / 2
                    wrist_mid_x = (current_points['left_wrist']['x'] + current_points['right_wrist']['x']) / 2
                    dx = wrist_mid_x - shoulder_mid_x
                    
                    if abs(dx) > 50:  # Threshold for significant horizontal movement
                        dynamics['direction'] = "rightward" if dx > 0 else "leftward"
                    else:
                        wrist_y = min(current_points['left_wrist']['y'], current_points['right_wrist']['y'])
                        shoulder_y = min(current_points['left_shoulder']['y'], current_points['right_shoulder']['y'])
                        dynamics['direction'] = "upward" if wrist_y < shoulder_y else "downward"
                
                # Classify swing speed based on velocity
                if swing_velocity > 100:  # Adjust thresholds based on your coordinate system
                    dynamics['speed'] = "powerful"
                elif swing_velocity > 50:
                    dynamics['speed'] = "quick"
                else:
                    dynamics['speed'] = "controlled"
                    
            elif sport_type in ['soccer', 'football']:
                # Analyze kick dynamics
                leg_joints = [
                    ('left', 'right'),  # For knees
                    ('right', 'left')   # For ankles
                ]
                kick_velocity = calculate_limb_velocity(
                    current_points, previous_points, leg_joints
                )
                
                # Classify kick force
                if kick_velocity > 80:
                    dynamics['force'] = "powerful"
                elif kick_velocity > 40:
                    dynamics['force'] = "firm"
                else:
                    dynamics['force'] = "precise"
                    
            elif sport_type == 'running':
                # Calculate stride length
                if all(k in current_points for k in ['left_ankle', 'right_ankle']):
                    stride_length = abs(
                        current_points['left_ankle']['x'] - 
                        current_points['right_ankle']['x']
                    )
                    
                    # Classify running speed based on stride
                    if stride_length > 100:
                        dynamics['speed'] = "sprinting"
                    elif stride_length > 60:
                        dynamics['speed'] = "running"
                    else:
                        dynamics['speed'] = "jogging"
        
        # Analyze general movement qualities
        if current_points:
            # Calculate body extension
            if all(k in current_points for k in ['left_shoulder', 'right_hip', 'right_knee']):
                torso_length = math.sqrt(
                    (current_points['right_hip']['x'] - current_points['left_shoulder']['x'])**2 +
                    (current_points['right_hip']['y'] - current_points['left_shoulder']['y'])**2
                )
                dynamics['extension'] = "extended" if torso_length > 150 else "compact"
            
            # Analyze balance
            if all(k in current_points for k in ['left_ankle', 'right_ankle']):
                stance_width = abs(
                    current_points['left_ankle']['x'] - 
                    current_points['right_ankle']['x']
                )
                dynamics['balance'] = "stable" if stance_width > 80 else "dynamic"
                
        return dynamics
        
    except Exception as e:
        print(f"Error in movement dynamics analysis: {str(e)}")
        return dynamics

def generate_movement_description(dynamics, sport_type):
    if not dynamics or not sport_type:
        return ""
        
    descriptions = []
    sport_type = sport_type.lower()
    
    # Add specific action description
    if dynamics['specific_action']:
        if sport_type == 'tennis':
            descriptions.append(f"executing a {dynamics['speed']} {dynamics['specific_action']}")
        elif sport_type == 'basketball':
            descriptions.append(f"performing a {dynamics['speed']} {dynamics['specific_action']}")
            
    # Add movement quality
    if dynamics['speed'] and dynamics['balance']:
        descriptions.append(f"with {dynamics['speed']} speed and {dynamics['balance']} balance")
        
    # Add directional information
    if dynamics['direction']:
        descriptions.append(f"moving {dynamics['direction']}")
        
    return " ".join(descriptions)

def get_pose_description(sport_type, pose_results):
    """Generate detailed descriptions of athlete poses based on pose analysis"""
    if not pose_results or 'poses' not in pose_results:
        return None

    descriptions = []
    pose_patterns = pose_results.get('pose_pattern')
    
    sport_poses = {
        'Cycling': {
            'cycling_position': [
                "with a streamlined forward-leaning posture",
                "maintaining an aerodynamic racing position",
                "showing perfect cycling form with a low profile",
                "demonstrating optimal riding technique"
            ],
            'default': [
                "displaying proper cycling form",
                "maintaining an efficient riding stance"
            ]
        },
        'Tennis': {
            'serve': [
                "executing a powerful serve with full extension",
                "reaching maximum height in the serving motion",
                "displaying perfect service technique",
                "positioning for an aggressive serve"
            ],
            'forehand': [
                "unleashing a dynamic forehand stroke",
                "rotating through a powerful forehand motion",
                "showing perfect forehand technique",
                "demonstrating exceptional racquet control"
            ],
            'backhand': [
                "executing a precise backhand stroke",
                "maintaining perfect balance during the backhand",
                "showing controlled backhand technique",
                "demonstrating expert backhand form"
            ],
            'default': [
                "displaying professional tennis form",
                "showing excellent court positioning"
            ]
        },
        'Baseball': {
            'batting': [
                "in perfect batting stance with eyes on the ball",
                "showing powerful hip rotation through the swing",
                "maintaining ideal contact position",
                "demonstrating textbook batting technique"
            ],
            'pitching': [
                "displaying perfect pitching mechanics",
                "executing the precise delivery motion",
                "showing masterful pitching form",
                "maintaining controlled throwing position"
            ],
            'default': [
                "showing professional baseball stance",
                "maintaining athletic readiness"
            ]
        },
        'Rugby': {
            'running': [
                "demonstrating powerful running form",
                "maintaining secure ball control while sprinting",
                "showing explosive acceleration technique",
                "displaying perfect ball-carrying form"
            ],
            'kicking': [
                "executing precise kicking technique",
                "showing perfect leg extension for the kick",
                "maintaining balanced kicking form",
                "demonstrating expert kicking mechanics"
            ],
            'default': [
                "displaying athletic rugby stance",
                "showing professional rugby positioning"
            ]
        }
    }

    # Get pose angles and positions from the first detected person
    if pose_results['poses']:
        first_pose = pose_results['poses'][0]
        
        # Get sport-specific description
        if sport_type in sport_poses:
            pose_options = sport_poses[sport_type].get(
                pose_patterns, 
                sport_poses[sport_type]['default']
            )
            descriptions.append(random.choice(pose_options))
            
            # Add specific angle-based descriptions
            if 'keypoints' in first_pose:
                pose_details = analyze_pose_details(first_pose['keypoints'], sport_type)
                if pose_details:
                    descriptions.append(pose_details)
                    
            # Add movement dynamics if available
            if 'movement_dynamics' in first_pose:
                dynamics = analyze_movement_dynamics(first_pose['movement_dynamics'], sport_type)
                if dynamics:
                    descriptions.append(dynamics)

    return ", ".join(descriptions) if descriptions else None

def get_enhanced_pose_description(sport_type_original, pose_results):
    """
    Get enhanced pose description with detailed error checking
    """
    print("DEBUG: Starting get_enhanced_pose_description")
    print(f"DEBUG: pose_results type: {type(pose_results)}")
    print(f"DEBUG: pose_results content: {pose_results}")

    try:
        # If pose_results is None or empty
        if not pose_results:
            print("DEBUG: pose_results is empty or None")
            return "maintaining athletic stance"

        # Get keypoints with validation
        keypoints = pose_results.get('keypoints', [])
        print(f"DEBUG: keypoints type: {type(keypoints)}")
        print(f"DEBUG: first keypoint sample: {keypoints[0] if keypoints else 'no keypoints'}")

        # Get pose details with validation
        pose_details = analyze_pose_details(keypoints, sport_type_original)
        print(f"DEBUG: pose_details type: {type(pose_details)}")
        print(f"DEBUG: pose_details content: {pose_details}")

        # Handle different return types
        if isinstance(pose_details, list):
            # Filter out None and empty strings, then join
            valid_details = [str(detail) for detail in pose_details if detail]
            return " ".join(valid_details) if valid_details else "maintaining athletic stance"
        elif isinstance(pose_details, str):
            return pose_details
        else:
            print(f"DEBUG: Unexpected pose_details type: {type(pose_details)}")
            return "maintaining athletic stance"

    except Exception as e:
        print(f"DEBUG: Error in get_enhanced_pose_description: {str(e)}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        return "maintaining athletic stance"

def generate_motion_description(dynamics, sport_type, form_analysis):
    """Generate more detailed motion descriptions"""
    try:
        motion_desc = []
        
        if dynamics.get('specific_action'):
            force = dynamics.get('force', 'controlled')
            motion_desc.append(f"executing a {force} {dynamics['specific_action']}")
        
        if form_analysis.get('body_alignment'):
            motion_desc.append(f"with {form_analysis['body_alignment']}")
            
        if dynamics.get('balance'):
            motion_desc.append(f"maintaining {dynamics['balance']} balance")
            
        if dynamics.get('speed'):
            motion_desc.append(f"at {dynamics['speed']} speed")
            
        return " ".join(motion_desc)
    except Exception as e:
        print(f"Error in generate_motion_description: {str(e)}")
        return ""

def interpret_emotion_in_context(emotion, sport_type, action_level):
    """Interpret emotions in the context of the sport and action"""
    if emotion == 'focus' and action_level > 0.7:
        return "displaying intense concentration"
    elif emotion == 'determination' and sport_type in ['boxing', 'martial_arts']:
        return "showing fierce competitive spirit"
    # Add more context-specific interpretations

def calculate_angle(point1, point2):
    """Calculate angle between two points relative to vertical"""
    if not point1 or not point2:
        return 0
    return abs(math.degrees(math.atan2(point2['y'] - point1['y'], 
                                     point2['x'] - point1['x'])))

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    if not point1 or not point2:
        return 0
    return math.sqrt((point2['x'] - point1['x'])**2 + 
                    (point2['y'] - point1['y'])**2)

def assess_pose_quality(keypoints, sport_type):
    """Assess overall quality of the pose"""
    if not keypoints:
        return None
        
    # Count number of detected keypoints
    detected_points = sum(1 for kp in keypoints if kp.get('confidence', 0) > 0.5)
    total_points = len(keypoints)
    
    if detected_points / total_points > 0.8:
        return "with excellent technical form"
    elif detected_points / total_points > 0.6:
        return "with good form"
    elif detected_points / total_points > 0.4:
        return "maintaining proper form"
    
    return None

def calculate_torso_angle(left_point, right_point):
    """Calculate torso angle relative to horizontal"""
    try:
        dx = right_point['x'] - left_point['x']
        dy = right_point['y'] - left_point['y']
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        return angle
    except:
        return 45

def calculate_arm_extension(points):
    """Calculate arm extension angle"""
    try:
        # Simplified calculation for example
        return 135
    except:
        return 90

def calculate_stride_length(points):
    """Calculate normalized stride length"""
    try:
        # Simplified calculation for example
        return 1.3
    except:
        return 1.0

def analyze_movement_dynamics(keypoints, prev_keypoints=None, sport_type=None):
    """
    Analyze movement dynamics from pose keypoints with detailed sport-specific analysis
    
    Args:
        keypoints: Current frame keypoints
        prev_keypoints: Previous frame keypoints (optional)
        sport_type: Type of sport being analyzed
    
    Returns:
        dict: Detailed movement analysis
    """
    dynamics = {
        'speed': None,
        'direction': None,
        'force': None,
        'rotation': None,
        'extension': None,
        'balance': None,
        'specific_action': None,
        'height': None,
        'stance': None
    }
    
    def calculate_angle(point1, point2, point3):
        """Calculate angle between three points"""
        try:
            if not all([point1, point2, point3]):
                return None
                
            vector1 = [point1['x'] - point2['x'], point1['y'] - point2['y']]
            vector2 = [point3['x'] - point2['x'], point3['y'] - point2['y']]
            
            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
            magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
            
            if magnitude1 == 0 or magnitude2 == 0:
                return None
                
            cos_angle = dot_product / (magnitude1 * magnitude2)
            angle = math.degrees(math.acos(max(-1, min(1, cos_angle))))
            return angle
        except:
            return None

    def find_keypoint(points, name):
        """Find keypoint by name in keypoint list"""
        try:
            if isinstance(points, list) and points and isinstance(points[0], (list, tuple)):
                # Array format
                keypoint_indices = {
                    'nose': 0,
                    'left_shoulder': 5, 'right_shoulder': 6,
                    'left_elbow': 7, 'right_elbow': 8,
                    'left_wrist': 9, 'right_wrist': 10,
                    'left_hip': 11, 'right_hip': 12,
                    'left_knee': 13, 'right_knee': 14,
                    'left_ankle': 15, 'right_ankle': 16
                }
                idx = keypoint_indices.get(name)
                if idx is not None and idx < len(points):
                    return {'x': float(points[idx][0]), 'y': float(points[idx][1])}
            else:
                # Dictionary format
                return next((kp for kp in points if kp.get('name') == name), None)
        except Exception as e:
            print(f"Error finding keypoint {name}: {str(e)}")
            return None

    try:
        if not keypoints:
            return dynamics
            
        # Get key points
        current_points = {}
        for joint in ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 
                     'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 
                     'right_hip', 'left_knee', 'right_knee', 'left_ankle', 
                     'right_ankle']:
            current_points[joint] = find_keypoint(keypoints, joint)

        # Basic stance analysis
        if all(current_points.get(joint) for joint in ['left_hip', 'right_hip', 'left_knee', 'right_knee']):
            hip_distance = math.sqrt(
                (current_points['right_hip']['x'] - current_points['left_hip']['x'])**2 +
                (current_points['right_hip']['y'] - current_points['left_hip']['y'])**2
            )
            knee_distance = math.sqrt(
                (current_points['right_knee']['x'] - current_points['left_knee']['x'])**2 +
                (current_points['right_knee']['y'] - current_points['left_knee']['y'])**2
            )
            
            if hip_distance > knee_distance * 1.2:
                dynamics['stance'] = 'wide'
            elif hip_distance < knee_distance * 0.8:
                dynamics['stance'] = 'narrow'
            else:
                dynamics['stance'] = 'balanced'

        # Sport-specific analysis
        if sport_type:
            sport_type = sport_type.lower()
            
            if sport_type in ['tennis', 'badminton']:
                # Analyze racquet sports movements
                if all(current_points.get(joint) for joint in ['right_shoulder', 'right_elbow', 'right_wrist']):
                    arm_angle = calculate_angle(
                        current_points['right_shoulder'],
                        current_points['right_elbow'],
                        current_points['right_wrist']
                    )
                    
                    # Determine specific tennis action
                    if current_points['right_wrist']['y'] < current_points['right_shoulder']['y']:
                        dynamics['specific_action'] = 'serve'
                        dynamics['height'] = 'overhead'
                    elif arm_angle and arm_angle < 90:
                        dynamics['specific_action'] = 'forehand'
                        dynamics['direction'] = 'forward'
                    else:
                        dynamics['specific_action'] = 'backhand'
                        dynamics['direction'] = 'lateral'

                    # Determine shot power
                    if arm_angle and arm_angle < 45:
                        dynamics['force'] = 'powerful'
                    elif arm_angle and arm_angle < 90:
                        dynamics['force'] = 'moderate'
                    else:
                        dynamics['force'] = 'controlled'
                        
            elif sport_type in ['basketball']:
                # Analyze basketball movements
                if all(current_points.get(joint) for joint in ['right_shoulder', 'right_elbow', 'right_wrist']):
                    arm_height = current_points['right_wrist']['y']
                    shoulder_height = current_points['right_shoulder']['y']
                    
                    if arm_height < shoulder_height:
                        dynamics['specific_action'] = 'shot'
                        dynamics['height'] = 'high'
                        dynamics['force'] = 'controlled'
                    else:
                        dynamics['specific_action'] = 'dribble'
                        dynamics['height'] = 'low'
                        dynamics['force'] = 'quick'
                        
            elif sport_type in ['soccer', 'football']:
                # Analyze soccer movements
                if all(current_points.get(joint) for joint in ['right_hip', 'right_knee', 'right_ankle']):
                    leg_angle = calculate_angle(
                        current_points['right_hip'],
                        current_points['right_knee'],
                        current_points['right_ankle']
                    )
                    
                    if leg_angle and leg_angle < 90:
                        dynamics['specific_action'] = 'kick'
                        dynamics['force'] = 'powerful'
                    else:
                        dynamics['specific_action'] = 'control'
                        dynamics['force'] = 'precise'
                        
            elif sport_type == 'running':
                # Analyze running form
                if all(current_points.get(joint) for joint in ['left_knee', 'right_knee', 'left_ankle', 'right_ankle']):
                    knee_height = min(current_points['left_knee']['y'], current_points['right_knee']['y'])
                    ankle_height = min(current_points['left_ankle']['y'], current_points['right_ankle']['y'])
                    
                    stride_length = abs(current_points['left_ankle']['x'] - current_points['right_ankle']['x'])
                    
                    if stride_length > 1.0:
                        dynamics['speed'] = 'sprinting'
                    elif stride_length > 0.7:
                        dynamics['speed'] = 'running'
                    else:
                        dynamics['speed'] = 'jogging'
                        
                    dynamics['form'] = 'high_knees' if knee_height < 0.5 else 'standard'
                    
            elif sport_type == 'cycling':
                # Analyze cycling position
                if all(current_points.get(joint) for joint in ['shoulder', 'hip', 'knee']):
                    torso_angle = calculate_angle(
                        current_points['shoulder'],
                        current_points['hip'],
                        current_points['knee']
                    )
                    
                    if torso_angle and torso_angle < 30:
                        dynamics['position'] = 'aero'
                        dynamics['speed'] = 'high'
                    elif torso_angle and torso_angle < 45:
                        dynamics['position'] = 'dropped'
                        dynamics['speed'] = 'moderate'
                    else:
                        dynamics['position'] = 'upright'
                        dynamics['speed'] = 'steady'

        # Calculate overall extension
        if all(current_points.get(joint) for joint in ['shoulder', 'hip', 'knee', 'ankle']):
            body_extension = abs(current_points['shoulder']['y'] - current_points['ankle']['y'])
            dynamics['extension'] = 'extended' if body_extension > 0.7 else 'compact'

        return dynamics
        
    except Exception as e:
        print(f"Error in analyze_movement_dynamics: {str(e)}")
        return dynamics

def analyze_pose_details(keypoints, sport_type):
    """
    Analyze pose details with enhanced debugging
    """
    print("DEBUG: Starting analyze_pose_details")
    print(f"DEBUG: keypoints type: {type(keypoints)}")
    print(f"DEBUG: sport_type: {sport_type}")

    try:
        if not keypoints:
            return ["maintaining athletic position"]

        details = []

        # Helper function with debug output
        def find_keypoint(name):
            try:
                if isinstance(keypoints[0], (list, tuple)):
                    print(f"DEBUG: Processing array-style keypoints for {name}")
                    keypoint_indices = {
                        'left_shoulder': 5,
                        'right_shoulder': 6,
                        'left_elbow': 7,
                        'right_elbow': 8,
                        'left_knee': 13,
                        'right_knee': 14
                    }
                    idx = keypoint_indices.get(name)
                    if idx is not None and idx < len(keypoints):
                        return {'x': float(keypoints[idx][0]), 'y': float(keypoints[idx][1])}
                else:
                    print(f"DEBUG: Processing dictionary-style keypoints for {name}")
                    return next((kp for kp in keypoints if kp.get('name') == name), None)
            except Exception as e:
                print(f"DEBUG: Error in find_keypoint for {name}: {str(e)}")
                return None

        # Get keypoints with debug output
        key_points = {
            'left_shoulder': find_keypoint('left_shoulder'),
            'right_shoulder': find_keypoint('right_shoulder'),
            'left_elbow': find_keypoint('left_elbow'),
            'right_elbow': find_keypoint('right_elbow'),
            'left_knee': find_keypoint('left_knee'),
            'right_knee': find_keypoint('right_knee')
        }

        print("DEBUG: Found keypoints:", {k: bool(v) for k, v in key_points.items()})

        # Basic stance
        details.append("maintaining athletic position")

        # Sport-specific analysis with validation
        sport_type_lower = str(sport_type).lower()
        
        # Add sport-specific details
        if sport_type_lower in ['tennis', 'badminton']:
            if all([key_points['left_shoulder'], key_points['right_shoulder'], 
                   key_points['left_elbow'], key_points['right_elbow']]):
                arm_height = min(float(key_points['left_shoulder']['y']), 
                               float(key_points['right_shoulder']['y']))
                details.append("executing a racket sport motion")

        # Always ensure we return a list of strings
        return [str(detail) for detail in details if detail]

    except Exception as e:
        print(f"DEBUG: Error in analyze_pose_details: {str(e)}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        return ["maintaining athletic position"]

def assess_alignment(left_shoulder, right_shoulder, left_knee, right_knee):
    """Helper function to assess body alignment"""
    try:
        shoulder_center = (
            (float(left_shoulder['x']) + float(right_shoulder['x']))/2, 
            (float(left_shoulder['y']) + float(right_shoulder['y']))/2
        )
        knee_center = (
            (float(left_knee['x']) + float(right_knee['x']))/2, 
            (float(left_knee['y']) + float(right_knee['y']))/2
        )
        
        alignment = 1.0 - min(abs(shoulder_center[0] - knee_center[0]), 0.3) / 0.3
        return max(0.0, min(1.0, alignment))
    except Exception as e:
        print(f"Error in assess_alignment: {str(e)}")
        return 0.6

#def assess_alignment(left_shoulder, right_shoulder, left_knee, right_knee):
#    """Helper function to assess body alignment"""
#    try:
#        shoulder_center = ((left_shoulder['x'] + right_shoulder['x'])/2, 
#                         (left_shoulder['y'] + right_shoulder['y'])/2)
#        knee_center = ((left_knee['x'] + right_knee['x'])/2, 
#                      (left_knee['y'] + right_knee['y'])/2)
#        
#        alignment = 1.0 - min(abs(shoulder_center[0] - knee_center[0]), 0.3) / 0.3
#        return max(0.0, min(1.0, alignment))
#    except:
#        return 0.6  # Default to moderate alignment if calculation fails

    return context

def analyze_formation_pattern(proximity_map, boxes):
    """Analyzes the spatial arrangement of athletes to determine formation patterns"""
    if not proximity_map or not boxes:
        return None
        
    # Calculate centroid distances and angles
    centroids = []
    for box in boxes:
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        centroids.append((x_center, y_center))
    
    # Analyze formation characteristics
    if len(centroids) < 3:
        return "linear"
        
    # Check for linear formation
    points = np.array(centroids)
    
    if len(points) >= 3:
        # Fit a line to the points
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Calculate distances to the fitted line
        distances = []
        for (x, y) in points:
            distance = abs((vy*(x - x0) - vx*(y - y0)) / np.sqrt(vx*vx + vy*vy))
            distances.append(distance)
            
        # If most points are close to the line, it's linear
        if np.mean(distances) < 0.2 * np.max(distances):
            return "linear"
    
    # Check for triangular formation
    if len(centroids) >= 3:
        hull = ConvexHull(points)
        if len(hull.vertices) == 3:
            return "triangular"
    
    # Check for circular formation
    if len(centroids) >= 4:
        center = np.mean(points, axis=0)
        distances_to_center = np.linalg.norm(points - center, axis=1)
        if np.std(distances_to_center) < 0.3 * np.mean(distances_to_center):
            return "circular"
    
    return "scattered"

def estimate_depth_layers(boxes):
    """Estimates the number of depth layers in the scene based on box positions"""
    if not boxes:
        return 1
        
    # Extract y-coordinates of box centers
    y_centers = [(box[1] + box[3]) / 2 for box in boxes]
    
    # Use clustering to identify distinct depth layers
    if len(y_centers) > 1:
        y_array = np.array(y_centers).reshape(-1, 1)
        clustering = KMeans(n_clusters=min(len(y_centers), 5)).fit(y_array)
        return len(np.unique(clustering.labels_))
    
    return 1

def analyze_athlete_stance(pose_results, sport_type):
    """Analyzes athlete stance based on pose keypoints"""
    if not pose_results or 'poses' not in pose_results:
        return "professional form"
        
    stance_descriptions = {
        'soccer': {
            'attacking': "aggressive attacking stance",
            'defensive': "solid defensive position",
            'running': "dynamic running form"
        },
        'tennis': {
            'serve': "perfect service stance",
            'forehand': "balanced forehand position",
            'backhand': "technical backhand form"
        },
        'basketball': {
            'shooting': "textbook shooting form",
            'defensive': "athletic defensive stance",
            'dribbling': "controlled dribbling position"
        }
    }
    
    sport_lower = sport_type.lower()
    if sport_lower in stance_descriptions:
        pose_type = pose_results.get('pose_pattern', 'default')
        return stance_descriptions[sport_lower].get(pose_type, "professional form")
    
    return "professional form"

def analyze_dual_athlete_interaction(pose_results, sport_type):
    """Analyzes interaction between two athletes"""
    interaction_types = {
        'tennis': "intense rally exchange",
        'soccer': "dynamic one-on-one contest",
        'basketball': "close defensive coverage",
        'rugby': "powerful physical contest"
    }
    
    return interaction_types.get(sport_type.lower(), "competitive exchange")

def analyze_group_movement(pose_results, sport_type):
    """Analyzes collective movement patterns"""
    movement_patterns = {
        'soccer': "coordinated tactical movement",
        'basketball': "fluid team rotation",
        'volleyball': "synchronized court coverage",
        'rugby': "cohesive unit movement"
    }
    
    return movement_patterns.get(sport_type.lower(), "synchronized movement")

def clean_caption(caption):
    """Cleans and formats the final caption"""
    # Remove multiple spaces
    caption = ' '.join(caption.split())
    
    # Fix punctuation spacing
    caption = caption.replace(' ,', ',')
    caption = caption.replace(' .', '.')
    caption = caption.replace('  ', ' ')
    
    # Ensure proper capitalization
    caption = caption[0].upper() + caption[1:]
    
    # Ensure proper ending
    if not caption.endswith('.'):
        caption += '.'
        
    # Add metadata
    timestamp = "2025-05-31 04:45:06"  # Using provided timestamp
    user = "Per0s"  # Using provided user login
    caption = f"{caption}\n\nGenerated by {user} at {timestamp} UTC"
    
    return caption

def generate_smart_suggestion(analysis_result):
    """
    Tạo 1 câu gợi ý thông minh dựa trên kết quả phân tích
    """
    # Lấy các thông số từ kết quả phân tích
    action_level = analysis_result.get('action_analysis', {}).get('action_level', 0)
    framing_quality = analysis_result.get('composition_analysis', {}).get('framing_quality', 'Unknown')
    athletes_count = analysis_result.get('detections', {}).get('athletes', 0)
    sport_type = analysis_result.get('composition_analysis', {}).get('sport_type', 'Unknown')

    # Tính điểm sharpness trung bình nếu có
    avg_sharpness = 0
    if 'sports_analysis' in analysis_result and 'sharpness_scores' in analysis_result['sports_analysis']:
        sharpness_scores = analysis_result['sports_analysis']['sharpness_scores']
        if sharpness_scores:
            avg_sharpness = sum(sharpness_scores) / len(sharpness_scores)

    # Kiểm tra có emotion không
    has_emotion = analysis_result.get('facial_analysis', {}).get('has_faces', False)
    emotion_intensity = analysis_result.get('facial_analysis', {}).get('emotion_intensity', 0) if has_emotion else 0

    # Ưu tiên theo mức độ quan trọng
    # 1. Action level thấp
    if action_level < 0.4:
        return "Try capturing during peak action moments for more dynamic sports photography."

    # 2. Framing kém
    if framing_quality in ['Poor', 'Could be improved', 'Fair']:
        return "Apply the rule of thirds and ensure subjects are well-positioned in the frame."

    # 3. Sharpness thấp
    if avg_sharpness < 0.5:
        return "Use faster shutter speed and proper focus to achieve sharper subject details."

    # 4. Không có emotion
    if not has_emotion and athletes_count > 0:
        return "Consider angles that capture athlete expressions for more engaging storytelling."

    # 5. Emotion tốt
    if has_emotion and emotion_intensity > 0.7:
        return "Excellent emotional capture! This adds great storytelling value to your sports photo."

    # 6. Action tốt nhưng có thể cải thiện khác
    if action_level > 0.7 and avg_sharpness > 0.6:
        return "Great action shot! Consider varying angles or including more context for visual variety."

    # 7. Suggestion chung theo môn thể thao
    if sport_type.lower() in ['soccer', 'football', 'basketball']:
        return "For team sports, try capturing player interactions and tactical moments."
    elif sport_type.lower() in ['tennis', 'golf', 'athletics']:
        return "Focus on technique and form - these sports offer great opportunities for skill showcase."

    # 8. Default suggestion
    return "Solid sports photography! Experiment with different perspectives to add creative flair."


def generate_high_action_intros(sport_name, scene_analysis, intro_context):
    """Generate introduction phrases for high-action scenes"""
    venue = intro_context.get('venue_type', 'standard')
    competition = intro_context.get('competition_level', 'professional')
    
    intros = [
        f"A spectacular moment of {sport_name} excellence",
        f"An electrifying display of {sport_name} athleticism",
        f"A breathtaking sequence in {sport_name} competition",
        f"An explosive {sport_name} action capture",
        f"A pinnacle moment in {sport_name} performance"
    ]
    
    if venue == 'professional':
        intros.extend([
            f"A professional-level {sport_name} spectacle",
            f"An elite {sport_name} competition moment"
        ])
        
    if competition == 'professional':
        intros.extend([
            f"A world-class display of {sport_name} mastery",
            f"A professional {sport_name} competition highlight"
        ])
        
    return intros

def generate_generic_high_action_intros(scene_analysis, intro_context):
    """Generate generic introduction phrases for high-action scenes"""
    return [
        "A powerful display of athletic excellence",
        "An intense moment of sporting brilliance",
        "A dynamic showcase of competitive spirit",
        "A spectacular athletic performance capture",
        "An extraordinary display of physical prowess"
    ]

def generate_medium_action_intros(sport_name, scene_analysis, intro_context):
    """Generate introduction phrases for medium-action scenes"""
    return [
        f"A skillful demonstration of {sport_name} technique",
        f"An engaging moment in {sport_name} competition",
        f"A well-executed {sport_name} sequence",
        f"A focused display of {sport_name} expertise",
        f"A technical showcase of {sport_name} ability"
    ]

def generate_generic_medium_action_intros(scene_analysis, intro_context):
    """Generate generic introduction phrases for medium-action scenes"""
    return [
        "A composed display of athletic skill",
        "A measured moment of sporting excellence",
        "A precise demonstration of competitive ability",
        "A focused athletic performance",
        "A controlled display of sporting expertise"
    ]

def generate_low_action_intros(sport_name, scene_analysis, intro_context):
    """Generate introduction phrases for low-action scenes"""
    return [
        f"A tactical moment in {sport_name} competition",
        f"A strategic pause during {sport_name} play",
        f"A calculated {sport_name} sequence",
        f"A preparatory phase of {sport_name} action",
        f"A measured moment in {sport_name} execution"
    ]

def generate_generic_low_action_intros(scene_analysis, intro_context):
    """Generate generic introduction phrases for low-action scenes"""
    return [
        "A moment of athletic preparation",
        "A strategic pause in competitive action",
        "A calculated sporting maneuver",
        "A focused pre-action sequence",
        "A deliberate competitive moment"
    ]

def enhance_intro_with_context(base_intro, scene_analysis, intro_context, sport_evidence):
    """Enhance the basic intro with additional context"""
    enhanced = base_intro
    
    # Add environmental context if available
    if intro_context.get('venue_type'):
        venue_desc = {
            'stadium': "in a packed stadium",
            'arena': "in a professional arena",
            'court': "on a competition court",
            'field': "on a professional field",
            'track': "on an athletic track"
        }
        venue = intro_context['venue_type'].lower()
        if venue in venue_desc:
            enhanced += f" {venue_desc[venue]}"
    
    # Add weather context for outdoor venues
    if intro_context.get('weather'):
        weather_desc = {
            'sunny': "under bright sunny conditions",
            'cloudy': "beneath dramatic cloud cover",
            'overcast': "in atmospheric overcast conditions",
            'indoor': "under controlled indoor conditions"
        }
        weather = intro_context['weather'].lower()
        if weather in weather_desc:
            enhanced += f" {weather_desc[weather]}"
    
    # Add competition level context if available
    if intro_context.get('competition_level') == 'professional':
        enhanced += " at the professional level"
    
    # Add time context if relevant
    if intro_context.get('time_of_day'):
        time_desc = {
            'morning': "during early morning competition",
            'afternoon': "in peak afternoon action",
            'evening': "under evening conditions",
            'night': "under professional lighting"
        }
        time = intro_context['time_of_day'].lower()
        if time in time_desc:
            enhanced += f" {time_desc[time]}"
    
    return enhanced

def get_sport_specific_description(sport_type, action_type):
    """
    Generate sport-specific movement descriptions based on sport type and action intensity.
    
    Args:
        sport_type (str): Type of sport being analyzed
        action_type (str): Level of action ('high' or 'medium')
    
    Returns:
        str: Sport-specific movement description
    """
    movement_descriptions = {
        'soccer': {
            'high': [
                "executing a precise ball control",
                "demonstrating exceptional footwork",
                "showing masterful ball handling",
                "performing an agile maneuver"
            ],
            'medium': [
                "maintaining tactical positioning",
                "showing controlled ball movement",
                "demonstrating steady footwork",
                "executing a calculated play"
            ]
        },
        'football': {
            'high': [
                "demonstrating explosive athleticism",
                "executing a powerful play",
                "showing remarkable agility",
                "performing an athletic leap"
            ],
            'medium': [
                "maintaining strategic position",
                "showing controlled movement",
                "executing a planned maneuver",
                "demonstrating tactical awareness"
            ]
        },
        'basketball': {
            'high': [
                "executing a dynamic play",
                "showing exceptional court movement",
                "demonstrating elite ball handling",
                "performing an explosive maneuver"
            ],
            'medium': [
                "maintaining court position",
                "showing controlled dribbling",
                "executing fundamental moves",
                "demonstrating tactical play"
            ]
        },
        'tennis': {
            'high': [
                "executing a powerful stroke",
                "demonstrating exceptional racquet control",
                "showing advanced court movement",
                "performing an aggressive return"
            ],
            'medium': [
                "maintaining steady form",
                "showing controlled strokes",
                "executing consistent returns",
                "demonstrating proper technique"
            ]
        },
        'baseball': {
            'high': [
                "executing a powerful swing",
                "demonstrating elite batting form",
                "showing exceptional fielding",
                "performing an explosive throw"
            ],
            'medium': [
                "maintaining batting stance",
                "showing controlled movement",
                "executing fundamental plays",
                "demonstrating proper form"
            ]
        },
        'swimming': {
            'high': [
                "executing powerful strokes",
                "demonstrating excellent technique",
                "showing exceptional speed",
                "performing an efficient turn"
            ],
            'medium': [
                "maintaining steady pace",
                "showing controlled breathing",
                "executing consistent strokes",
                "demonstrating proper form"
            ]
        },
        'volleyball': {
            'high': [
                "executing a powerful spike",
                "demonstrating exceptional jumping",
                "showing advanced court coverage",
                "performing an aggressive serve"
            ],
            'medium': [
                "maintaining court position",
                "showing controlled movements",
                "executing basic plays",
                "demonstrating proper form"
            ]
        },
        'track': {
            'high': [
                "executing powerful strides",
                "demonstrating explosive speed",
                "showing exceptional form",
                "performing at peak acceleration"
            ],
            'medium': [
                "maintaining steady pace",
                "showing controlled strides",
                "executing consistent form",
                "demonstrating proper technique"
            ]
        },
        'running': {
            'high': [
                "executing powerful strides",
                "demonstrating peak form",
                "showing excellent pace",
                "performing at high intensity"
            ],
            'medium': [
                "maintaining steady rhythm",
                "showing consistent form",
                "executing controlled strides",
                "demonstrating proper technique"
            ]
        },
        'boxing': {
            'high': [
                "executing powerful combinations",
                "demonstrating explosive movement",
                "showing advanced footwork",
                "performing aggressive strikes"
            ],
            'medium': [
                "maintaining guard position",
                "showing controlled movement",
                "executing basic combinations",
                "demonstrating proper form"
            ]
        },
        'skiing': {
            'high': [
                "executing advanced techniques",
                "demonstrating exceptional control",
                "showing masterful edge control",
                "performing aggressive turns"
            ],
            'medium': [
                "maintaining balanced form",
                "showing controlled descent",
                "executing steady turns",
                "demonstrating proper technique"
            ]
        },
        'skating': {
            'high': [
                "executing complex maneuvers",
                "demonstrating exceptional balance",
                "showing advanced footwork",
                "performing precise jumps"
            ],
            'medium': [
                "maintaining steady rhythm",
                "showing controlled movements",
                "executing basic elements",
                "demonstrating proper form"
            ]
        },
        'surfing': {
            'high': [
                "executing advanced maneuvers",
                "demonstrating exceptional control",
                "showing masterful wave reading",
                "performing aggressive turns"
            ],
            'medium': [
                "maintaining balance",
                "showing controlled riding",
                "executing basic maneuvers",
                "demonstrating proper form"
            ]
        },
        'skateboarding': {
            'high': [
                "executing complex tricks",
                "demonstrating exceptional control",
                "showing advanced technique",
                "performing aggressive maneuvers"
            ],
            'medium': [
                "maintaining balance",
                "showing controlled movement",
                "executing basic tricks",
                "demonstrating proper form"
            ]
        },
        'golf': {
            'high': [
                "executing a powerful swing",
                "demonstrating exceptional control",
                "showing advanced technique",
                "performing precise shots"
            ],
            'medium': [
                "maintaining proper stance",
                "showing controlled swing",
                "executing basic shots",
                "demonstrating proper form"
            ]
        },
        'rugby': {
            'high': [
                "executing powerful runs",
                "demonstrating explosive movement",
                "showing aggressive tackles",
                "performing dynamic plays"
            ],
            'medium': [
                "maintaining tactical position",
                "showing controlled movement",
                "executing set plays",
                "demonstrating proper form"
            ]
        },
        'martial arts': {
            'high': [
                "executing complex techniques",
                "demonstrating explosive movement",
                "showing advanced combinations",
                "performing powerful strikes"
            ],
            'medium': [
                "maintaining proper stance",
                "showing controlled movement",
                "executing basic techniques",
                "demonstrating proper form"
            ]
        },
        'cycling': {
            'high': [
                "maintaining optimal racing position",
                "demonstrating powerful pedaling",
                "showing exceptional endurance",
                "performing strategic acceleration"
            ],
            'medium': [
                "maintaining steady cadence",
                "showing controlled riding",
                "executing proper form",
                "demonstrating efficient technique"
            ]
        }
    }

    sport_type_lower = sport_type.lower()
    if sport_type_lower in movement_descriptions:
        action_level = 'high' if action_type == 'high' else 'medium'
        return random.choice(movement_descriptions[sport_type_lower][action_level])
    
    # Default return for unknown sports
    return "displaying athletic movement" if action_type == 'high' else "maintaining proper form"

def verify_sport_movement(sport_type, movement_type):
    """
    Verify if a movement is valid for a specific sport.
    
    Args:
        sport_type (str): Type of sport being analyzed
        movement_type (str): Type of movement being verified
    
    Returns:
        bool: Whether the movement is valid for the sport
    """
    sport_valid_movements = {
        'soccer': ['kick', 'run', 'jump', 'header', 'tackle', 'dribble'],
        'football': ['run', 'jump', 'catch', 'throw', 'tackle', 'block'],
        'basketball': ['jump', 'run', 'dribble', 'shoot', 'pass', 'block'],
        'tennis': ['serve', 'forehand', 'backhand', 'volley', 'smash', 'run'],
        'baseball': ['swing', 'throw', 'catch', 'run', 'slide', 'pitch'],
        'swimming': ['freestyle', 'butterfly', 'backstroke', 'breaststroke', 'dive', 'turn'],
        'volleyball': ['jump', 'spike', 'serve', 'block', 'dig', 'set'],
        'track': ['sprint', 'run', 'jump', 'hurdle', 'relay', 'start'],
        'running': ['sprint', 'jog', 'run', 'stride', 'accelerate', 'pace'],
        'boxing': ['punch', 'block', 'dodge', 'weave', 'clinch', 'footwork'],
        'skiing': ['carve', 'jump', 'traverse', 'turn', 'stop', 'descend'],
        'skating': ['jump', 'spin', 'glide', 'turn', 'stop', 'stride'],
        'surfing': ['ride', 'turn', 'cutback', 'aerial', 'paddle', 'duck dive'],
        'skateboarding': ['ollie', 'grind', 'flip', 'slide', 'jump', 'turn'],
        'golf': ['swing', 'putt', 'chip', 'drive', 'approach', 'pitch'],
        'rugby': ['run', 'pass', 'tackle', 'kick', 'ruck', 'maul'],
        'martial arts': ['kick', 'punch', 'block', 'throw', 'grapple', 'strike'],
        'cycling': ['pedal', 'sprint', 'climb', 'descend', 'accelerate', 'draft']
    }
    
    return (sport_type.lower() in sport_valid_movements and 
            movement_type.lower() in sport_valid_movements[sport_type.lower()])

def generate_sports_caption(analysis_result):
    """
    Generates high-quality, natural-sounding English captions for sports images based on analysis results.

    Special handling for:
    - Groups of athletes (>6 athletes with >3 objects close together)
    - Main athlete highlight when standing alone or with minimal overlap

    Args:
        analysis_result: Dictionary containing sports image analysis data

    Returns:
        str: Well-crafted caption describing the sports image
    """
    
    # Check if analysis_result is provided
    if analysis_result is None:
        return "Error: No image analysis data available"

    # Initialize default values for required fields
    default_analysis = {
        'detections': {},
        'sports_analysis': {},
        'action_analysis': {},
        'composition_analysis': {},
        'facial_analysis': {}
    }
    
    # Merge provided analysis with defaults
    analysis_result = {**default_analysis, **analysis_result}
    
    # Extract key information with safe gets
    detections = analysis_result.get('detections', {})
    sports_analysis = analysis_result.get('sports_analysis', {})
    action_analysis = analysis_result.get('action_analysis', {})
    composition_analysis = analysis_result.get('composition_analysis', {})
    facial_analysis = analysis_result.get('facial_analysis', {})

    # Add error handling for required fields
    if not all([detections, sports_analysis, action_analysis, composition_analysis]):
        return "Error: Incomplete image analysis data"
    
    # Extract key information from analysis results
    detections = analysis_result.get('detections', {})
    sports_analysis = analysis_result.get('sports_analysis', {})
    action_analysis = analysis_result.get('action_analysis', {})
    composition_analysis = analysis_result.get('composition_analysis', {})
    facial_analysis = analysis_result.get('facial_analysis', {})

    # Initialize caption component parts
    intro_phrases = []  # Opening sentence
    subject_phrases = []  # Main subject description
    action_phrases = []  # Action description
    emotion_phrases = []  # Emotional aspects
    detail_phrases = []  # Additional details
    closing_phrases = []  # Conclusion
    
    # ----------------- 1. IDENTIFY SPORT TYPE -----------------
    sport_type = composition_analysis.get('sport_type', 'Running').lower()
    sport_type_original = composition_analysis.get('sport_type', 'Running')
    
    # Add the try-except block here, around the pose analysis
    # Replace the section in your try block with this:
    try:
        pose_results = analysis_result.get('pose_analysis', {})
        print(f"DEBUG: Main - pose_results: {pose_results}")
        
        # Extract keypoints correctly from pose_results structure
        current_keypoints = []
        if 'poses' in pose_results and pose_results['poses']:
            current_keypoints = pose_results['poses'][0].get('keypoints', [])
        previous_keypoints = pose_results.get('previous_keypoints', [])
        
        # Get sport type and action level
        detected_sport = pose_results.get('identified_sport', sport_type_original)
        action_level = action_analysis.get('action_level', 0.5)
        
        # Determine action quality based on action level
        if action_level > 0.7:
            action_quality = 'High'
        elif action_level > 0.4:
            action_quality = 'Medium'
        else:
            action_quality = 'Low'
        
        print(f"DEBUG: action_level = {action_level}")
        print(f"DEBUG: action_quality = {action_quality}")
        
        # 1. Analyze athlete stance first
        stance_description = analyze_athlete_stance(pose_results, detected_sport)
        
        # 2. Get movement dynamics
        movement_dynamics = analyze_movement_dynamics(
            current_keypoints,
            previous_keypoints,
            sport_type_original
        )
        print(f"DEBUG: Movement dynamics: {movement_dynamics}")
        
        # 3. Analyze pose details
        pose_details = analyze_pose_details(current_keypoints, detected_sport)
        
        # 4. Generate motion description
        form_analysis = {
            'body_alignment': stance_description,
            'movement_quality': pose_details[0] if pose_details else "maintaining form"
        }
        
        motion_desc = generate_motion_description(
            movement_dynamics,
            detected_sport,
            form_analysis
        )
        
        # 5. Get emotion interpretation
        emotion_context = interpret_emotion_in_context(
            facial_analysis.get('dominant_emotion', ''),
            detected_sport,
            action_level
        )
        
        # 6. Build the action description
        action_desc = []
        
        # Add stance and pose details
        action_desc.append(stance_description)
        
        # Add sport-specific description
        sport_specific_action = get_sport_specific_description(
            detected_sport, 
            'high' if action_level > 0.7 else 'medium'
        )
        if sport_specific_action:
            action_desc.append(sport_specific_action)
        
        # Add motion description
        if motion_desc:
            action_desc.append(motion_desc)
        
        # Add emotion context
        if emotion_context:
            action_desc.append(emotion_context)
        
        # Combine all descriptions
        final_action_desc = " ".join(action_desc)
        action_phrases.append(final_action_desc)
        
        # ----------------- 5. DESCRIBE ACTION -----------------
        if action_quality:
            # Get the base action description from our previous analysis
            base_action_desc = action_phrases[-1] if action_phrases else ""
            
            # Create sport-specific action options
            if detected_sport and base_action_desc:
                if action_quality == 'High':
                    if detected_sport in ['soccer', 'football', 'basketball', 'volleyball']:
                        action_options = [
                            f"{base_action_desc}, intensifying during a crucial play",
                            f"{base_action_desc}, peaking at a decisive moment",
                            f"{base_action_desc}, executing with exceptional precision",
                            f"{base_action_desc}, demonstrating elite-level execution"
                        ]
                    elif detected_sport in ['tennis', 'baseball', 'golf']:
                        action_options = [
                            f"{base_action_desc}, achieving perfect technical form",
                            f"{base_action_desc}, displaying masterful control",
                            f"{base_action_desc}, executing with pinpoint accuracy",
                            f"{base_action_desc}, showing expert timing"
                        ]
                    elif detected_sport in ['skiing', 'snowboarding']:
                        action_options = [
                            f"{base_action_desc}, mastering the challenging conditions",
                            f"{base_action_desc}, maintaining perfect balance",
                            f"{base_action_desc}, showing exceptional terrain reading",
                            f"{base_action_desc}, demonstrating advanced technique"
                        ]
                    elif detected_sport in ['running', 'track', 'swimming']:
                        action_options = [
                            f"{base_action_desc}, reaching peak performance",
                            f"{base_action_desc}, displaying supreme conditioning",
                            f"{base_action_desc}, maintaining optimal form",
                            f"{base_action_desc}, showing exceptional endurance"
                        ]
                    else:
                        action_options = [
                            f"{base_action_desc}, reaching athletic excellence",
                            f"{base_action_desc}, showing professional-level execution",
                            f"{base_action_desc}, demonstrating superior skill",
                            f"{base_action_desc}, displaying technical mastery"
                        ]
                elif action_quality == 'Medium':
                    action_options = [
                        f"{base_action_desc}, showing good technical execution",
                        f"{base_action_desc}, maintaining consistent form",
                        f"{base_action_desc}, displaying solid athleticism",
                        f"{base_action_desc}, demonstrating proper technique"
                    ]
                else:
                    action_options = [
                        f"{base_action_desc}, focusing on fundamentals",
                        f"{base_action_desc}, maintaining basic form",
                        f"{base_action_desc}, showing controlled movement",
                        f"{base_action_desc}, executing with care"
                    ]

                # Replace the previous action description with the enhanced version
                action_phrases[-1] = random.choice(action_options)
        
        # Ensure string output for pose description
        if isinstance(pose_description, list):
            pose_description = " ".join(str(x) for x in pose_description if x)
        elif not isinstance(pose_description, str):
            pose_description = str(pose_description)
                
    except Exception as e:
        print(f"Error analyzing pose: {str(e)}")
        print(f"DEBUG: Exception details - {type(e).__name__}")
        pose_description = "maintaining athletic stance"
        
    athlete_count = detections.get('athletes', 0)
    action_level = action_analysis.get('action_level', 0)
    equipment = action_analysis.get('equipment_types', [])

    # Sport-specific terminology
    sport_specific_terms = {
    'soccer': {
        'positions': ['striker', 'defender', 'midfielder', 'goalkeeper'],
        'techniques': ['dribbling', 'passing', 'shooting', 'tackling'],
        'equipment': ['soccer ball', 'cleats', 'shin guards'],
        'actions': ['kick', 'header', 'slide tackle', 'save'],
        'field_terms': ['pitch', 'penalty area', 'goal', 'sideline']
    },
    'football': {
        'positions': ['quarterback', 'receiver', 'linebacker', 'safety'],
        'techniques': ['passing', 'catching', 'blocking', 'rushing'],
        'equipment': ['football', 'helmet', 'pads'],
        'actions': ['throw', 'catch', 'tackle', 'run'],
        'field_terms': ['end zone', 'yard line', 'sideline', 'stadium']
    },
    'basketball': {
        'positions': ['guard', 'forward', 'center'],
        'techniques': ['dribbling', 'shooting', 'passing', 'defense'],
        'equipment': ['basketball', 'court shoes'],
        'actions': ['shoot', 'dribble', 'pass', 'block'],
        'court_terms': ['court', 'hoop', 'backboard', 'three-point line']
    },
    'tennis': {
        'positions': ['baseline', 'net', 'serving position'],
        'techniques': ['serve', 'forehand', 'backhand', 'volley'],
        'equipment': ['tennis racket', 'tennis ball', 'court shoes'],
        'actions': ['serve', 'return', 'smash', 'lob'],
        'court_terms': ['court', 'baseline', 'service box', 'net']
    },
    'baseball': {
        'positions': ['pitcher', 'batter', 'catcher', 'fielder'],
        'techniques': ['pitching', 'batting', 'catching', 'fielding'],
        'equipment': ['baseball', 'bat', 'glove', 'helmet'],
        'actions': ['pitch', 'swing', 'catch', 'throw'],
        'field_terms': ['diamond', 'base', 'mound', 'outfield']
    },
    'swimming': {
        'positions': ['starting block', 'in-water position'],
        'techniques': ['freestyle', 'butterfly', 'backstroke', 'breaststroke'],
        'equipment': ['swimsuit', 'goggles', 'swim cap'],
        'actions': ['dive', 'stroke', 'turn', 'finish'],
        'pool_terms': ['lane', 'pool', 'wall', 'starting block']
    },
    'volleyball': {
        'positions': ['setter', 'spiker', 'libero'],
        'techniques': ['serving', 'setting', 'spiking', 'blocking'],
        'equipment': ['volleyball', 'knee pads'],
        'actions': ['serve', 'spike', 'block', 'dig'],
        'court_terms': ['court', 'net', 'line', 'antenna']
    },
    'track': {
        'positions': ['starting blocks', 'lanes'],
        'techniques': ['sprinting', 'pacing', 'passing'],
        'equipment': ['spikes', 'starting blocks', 'baton'],
        'actions': ['sprint', 'hand-off', 'stride', 'finish'],
        'track_terms': ['lane', 'finish line', 'relay zone', 'curve']
    },
    'running': {
        'positions': ['starting position', 'racing position'],
        'techniques': ['sprinting', 'distance running', 'trail running'],
        'equipment': ['running shoes', 'track spikes', 'racing kit'],
        'actions': ['sprint', 'stride', 'accelerate', 'pace'],
        'terms': ['track', 'trail', 'course', 'finish line']
    },
    'boxing': {
        'positions': ['fight stance', 'guard position'],
        'techniques': ['jab', 'cross', 'hook', 'uppercut'],
        'equipment': ['gloves', 'headgear', 'mouthguard'],
        'actions': ['punch', 'block', 'dodge', 'clinch'],
        'ring_terms': ['ring', 'corner', 'ropes', 'canvas']
    },
    'skiing': {
        'positions': ['racing stance', 'carving position'],
        'techniques': ['carving', 'parallel turns', 'moguls'],
        'equipment': ['skis', 'poles', 'boots', 'goggles'],
        'actions': ['carve', 'jump', 'traverse', 'brake'],
        'slope_terms': ['slope', 'powder', 'trail', 'mogul']
    },
    'skating': {
        'positions': ['racing position', 'artistic pose'],
        'techniques': ['jumps', 'spins', 'footwork', 'gliding'],
        'equipment': ['skates', 'protective gear'],
        'actions': ['jump', 'spin', 'glide', 'turn'],
        'rink_terms': ['ice', 'rink', 'barrier', 'center']
    },
    'surfing': {
        'positions': ['standing', 'paddling', 'duck diving'],
        'techniques': ['carving', 'aerial', 'tube riding'],
        'equipment': ['surfboard', 'wetsuit', 'leash'],
        'actions': ['paddle', 'pop-up', 'carve', 'ride'],
        'ocean_terms': ['wave', 'break', 'lineup', 'beach']
    },
    'skateboarding': {
        'positions': ['regular stance', 'goofy stance'],
        'techniques': ['ollie', 'kickflip', 'grind'],
        'equipment': ['skateboard', 'helmet', 'pads'],
        'actions': ['push', 'flip', 'grind', 'slide'],
        'park_terms': ['ramp', 'rail', 'halfpipe', 'street']
    },
    'golf': {
        'positions': ['address position', 'stance'],
        'techniques': ['drive', 'approach', 'putting'],
        'equipment': ['clubs', 'golf ball', 'tees'],
        'actions': ['swing', 'putt', 'chip', 'drive'],
        'course_terms': ['green', 'fairway', 'bunker', 'rough']
    },
    'rugby': {
        'positions': ['forward', 'back', 'scrum-half'],
        'techniques': ['passing', 'tackling', 'kicking'],
        'equipment': ['rugby ball', 'scrum cap', 'mouthguard'],
        'actions': ['pass', 'tackle', 'ruck', 'maul'],
        'field_terms': ['try line', 'pitch', 'in-goal area', 'halfway']
    },
    'martial arts': {
        'positions': ['fighting stance', 'guard position'],
        'techniques': ['strikes', 'kicks', 'grappling'],
        'equipment': ['gi', 'belt', 'protective gear'],
        'actions': ['strike', 'block', 'throw', 'grapple'],
        'mat_terms': ['mat', 'ring', 'dojo', 'training area']
    }
}
    # Determine sport type based on name and equipment
    detected_sport = 'unknown'

    for sport_name, terms in sport_specific_terms.items():
        # Check sport name
        if sport_name in sport_type.lower():
            detected_sport = sport_name
            break

        # Check equipment
        for eq in equipment:
            eq_lower = eq.lower()
            if sport_name in eq_lower or any(term.lower() in eq_lower for term in terms):
                detected_sport = sport_name
                break

    # If no specific sport found but "sports ball" is detected
    if detected_sport == 'unknown' and any('ball' in eq.lower() for eq in equipment):
        detected_sport = 'ball sport'

    # ----------------- 2. ANALYZE SCENE COMPLEXITY -----------------
    # Check if it's a crowded scene (many athletes close together)
    is_crowded_scene = False
    has_clear_main_subject = False

    # Determine if scene is crowded (>6 athletes and >3 objects close together)
    if athlete_count > 6 and len(detections.get('boxes', [])) > 3:
        # Check proximity of objects if we have position data
        if 'key_subjects' in sports_analysis and len(sports_analysis['key_subjects']) > 3:
            # Count objects that are close to each other
            overlapping_count = 0
            boxes = [subj['box'] for subj in sports_analysis['key_subjects'] if 'box' in subj]

            # Simple overlap detection by checking box proximity
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    # Calculate centers
                    x1_center = (boxes[i][0] + boxes[i][2]) / 2
                    y1_center = (boxes[i][1] + boxes[i][3]) / 2
                    x2_center = (boxes[j][0] + boxes[j][2]) / 2
                    y2_center = (boxes[j][1] + boxes[j][3]) / 2

                    # Calculate distance between centers
                    distance = ((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2) ** 0.5

                    # Define "close" as less than average box width/height
                    avg_width = ((boxes[i][2] - boxes[i][0]) + (boxes[j][2] - boxes[j][0])) / 2
                    avg_height = ((boxes[i][3] - boxes[i][1]) + (boxes[j][3] - boxes[j][1])) / 2
                    avg_size = (avg_width + avg_height) / 2

                    if distance < avg_size:
                        overlapping_count += 1

            # If many overlaps detected, consider it a crowded scene
            is_crowded_scene = overlapping_count > 3

    # Determine if there's a clear main subject
    if 'key_subjects' in sports_analysis and sports_analysis['key_subjects']:
        main_subject = sports_analysis['key_subjects'][0]
        # Check if the main subject is significantly more prominent
        if len(sports_analysis['key_subjects']) > 1:
            second_subject = sports_analysis['key_subjects'][1]
            if main_subject.get('prominence', 0) > second_subject.get('prominence', 0) * 1.5:
                has_clear_main_subject = True
        else:
            has_clear_main_subject = True

    # ----------------- 3. CREATE INTRO PHRASES -----------------
    import random

    if action_level > 0.7:  # High action
        if detected_sport != 'unknown':
            sport_name = sport_type_original
            intro_options = [
                f"A dramatic moment in {sport_name}",
                f"An intense action shot from {sport_name}",
                f"A thrilling {sport_name} action capture",
                f"A peak moment in {sport_name} competition",
                f"A powerful {sport_name} action frame"
            ]
        else:
            intro_options = [
                "A stunning sports action moment",
                "An energetic capture from a sporting event",
                "A dynamic shot showcasing athletic prowess",
                "An impressive display of sports action",
                "A high-intensity athletic moment"
            ]
    elif action_level > 0.3:  # Medium action
        if detected_sport != 'unknown':
            sport_name = sport_type_original
            intro_options = [
                f"An engaging {sport_name} moment",
                f"A skillful display in {sport_name}",
                f"An active {sport_name} sequence",
                f"A competitive {sport_name} scene",
                f"A focused moment during {sport_name} play"
            ]
        else:
            intro_options = [
                "A captivating sports moment",
                "An active scene from a sporting event",
                "A dynamic athletic display",
                "A moment of sporting engagement",
                "A vibrant athletic performance"
            ]
    else:  # Low action
        if detected_sport != 'unknown':
            sport_name = sport_type_original
            intro_options = [
                f"A composed moment in {sport_name}",
                f"A strategic pause during {sport_name}",
                f"A reflective scene from {sport_name}",
                f"A calm moment in {sport_name} activity",
                f"A preparation phase for {sport_name} action"
            ]
        else:
            intro_options = [
                "A contemplative sports moment",
                "A composed athletic scene",
                "A moment of focus in sports",
                "A strategic pause in athletic activity",
                "A calm moment in sports performance"
            ]

    intro_phrases.append(random.choice(intro_options))

    # ----------------- 4. DESCRIBE SUBJECTS -----------------
    # Consider the special cases for athlete descriptions
    if is_crowded_scene:
        # For crowded scenes with many athletes and objects
        subject_options = [
            "featuring a group of athletes in close formation",
            "showcasing a cluster of competitors in action",
            "capturing a tight group of players in motion",
            "highlighting the coordinated movement of multiple athletes",
            "depicting a formation of athletes in synchronized action"
        ]
        subject_phrases.append(random.choice(subject_options))
    elif has_clear_main_subject and athlete_count > 1:
        # For scenes with a clear main subject and other athletes
        if 'key_subjects' in sports_analysis and sports_analysis['key_subjects']:
            main_class = sports_analysis['key_subjects'][0].get('class', 'athlete')
            if main_class.lower() == 'person':
                main_desc = "athlete"
            else:
                main_desc = main_class

            subject_options = [
                f"featuring a standout {main_desc} among {athlete_count - 1} other competitors",
                f"focusing on a primary {main_desc} with {athlete_count - 1} other athletes visible",
                f"highlighting the main performer against a backdrop of {athlete_count - 1} others",
                f"showcasing a central {main_desc} with supporting participants",
                f"centered on the primary athlete with additional performers in frame"
            ]
            subject_phrases.append(random.choice(subject_options))
    elif athlete_count > 0:
        # Standard description based on athlete count
        if athlete_count == 1:
            subject_options = [
                "featuring a solo athlete",
                "showcasing an individual competitor",
                "capturing a lone performer",
                "highlighting a single athlete's form",
                "focusing on an individual's athletic prowess"
            ]
        elif athlete_count == 2:
            subject_options = [
                f"featuring a pair of athletes in competition",
                f"showcasing two competitors facing off",
                f"capturing the dynamic between two athletes",
                f"highlighting the interaction of two competitors"
            ]
        else:
            subject_options = [
                f"featuring {athlete_count} athletes in action",
                f"showcasing a team of {athlete_count} competitors",
                f"capturing the coordination of {athlete_count} athletes",
                f"highlighting multiple performers in synchronized motion"
            ]

        subject_phrases.append(random.choice(subject_options))


    # ----------------- 5. DESCRIBE ACTION -----------------
    if action_quality:
        # Get the base action description from our previous analysis
        base_action_desc = action_phrases[-1] if action_phrases else ""
        
        # Create sport-specific action options
        if detected_sport and base_action_desc:
            if action_quality == 'High':
                if detected_sport in ['soccer', 'football', 'basketball', 'volleyball']:
                    action_options = [
                        f"{base_action_desc}, intensifying during a crucial play",
                        f"{base_action_desc}, peaking at a decisive moment",
                        f"{base_action_desc}, executing with exceptional precision",
                        f"{base_action_desc}, demonstrating elite-level execution"
                    ]
                elif detected_sport in ['tennis', 'baseball', 'golf']:
                    action_options = [
                        f"{base_action_desc}, achieving perfect technical form",
                        f"{base_action_desc}, displaying masterful control",
                        f"{base_action_desc}, executing with pinpoint accuracy",
                        f"{base_action_desc}, showing expert timing"
                    ]
                elif detected_sport in ['skiing', 'snowboarding']:
                    action_options = [
                        f"{base_action_desc}, mastering the challenging conditions",
                        f"{base_action_desc}, maintaining perfect balance",
                        f"{base_action_desc}, showing exceptional terrain reading",
                        f"{base_action_desc}, demonstrating advanced technique"
                    ]
                elif detected_sport in ['running', 'track', 'swimming']:
                    action_options = [
                        f"{base_action_desc}, reaching peak performance",
                        f"{base_action_desc}, displaying supreme conditioning",
                        f"{base_action_desc}, maintaining optimal form",
                        f"{base_action_desc}, showing exceptional endurance"
                    ]
                else:
                    action_options = [
                        f"{base_action_desc}, reaching athletic excellence",
                        f"{base_action_desc}, showing professional-level execution",
                        f"{base_action_desc}, demonstrating superior skill",
                        f"{base_action_desc}, displaying technical mastery"
                    ]
            elif action_quality == 'Medium':
                action_options = [
                    f"{base_action_desc}, showing good technical execution",
                    f"{base_action_desc}, maintaining consistent form",
                    f"{base_action_desc}, displaying solid athleticism",
                    f"{base_action_desc}, demonstrating proper technique"
                ]
            else:
                action_options = [
                    f"{base_action_desc}, focusing on fundamentals",
                    f"{base_action_desc}, maintaining basic form",
                    f"{base_action_desc}, showing controlled movement",
                    f"{base_action_desc}, executing with care"
                ]

            # Replace the previous action description with the enhanced version
            action_phrases[-1] = random.choice(action_options)

    # ----------------- 6. DESCRIBE EQUIPMENT -----------------
    if equipment:
        eq_list = []
        for eq in equipment:
            if eq.lower() == "sports ball":
                if detected_sport == "soccer":
                    eq_list.append("a soccer ball")
                elif detected_sport == "basketball":
                    eq_list.append("a basketball")
                elif detected_sport == "volleyball":
                    eq_list.append("a volleyball")
                elif detected_sport == "tennis":
                    eq_list.append("a tennis ball")
                else:
                    eq_list.append("a sports ball")
            elif eq.lower() == "tennis racket":
                eq_list.append("a tennis racket")
            else:
                eq_list.append(eq)

        if len(eq_list) == 1:
            detail_phrases.append(f"with {eq_list[0]}")
        elif len(eq_list) == 2:
            detail_phrases.append(f"with {eq_list[0]} and {eq_list[1]}")
        elif len(eq_list) > 2:
            detail_phrases.append(f"with {', '.join(eq_list[:-1])}, and {eq_list[-1]}")

    # ----------------- 7. DESCRIBE EMOTION -----------------
    if facial_analysis and facial_analysis.get('has_faces', False):
        emotion = facial_analysis.get('dominant_emotion', '').lower()
        intensity = facial_analysis.get('emotion_intensity', 0)

        if intensity > 0.5:  # Clear emotion
            if emotion == 'happy' or emotion == 'happiness':
                emotion_options = [
                    "displaying evident joy on their face",
                    "with a confident smile radiating triumph",
                    "revealing excitement through their expression",
                    "with a remarkably upbeat expression"
                ]
            elif emotion == 'neutral':
                emotion_options = [
                    "maintaining intense concentration",
                    "showing steely determination in their gaze",
                    "keeping a professionally composed expression",
                    "with unwavering focus visible in their eyes"
                ]
            elif emotion == 'sad' or emotion == 'sadness':
                emotion_options = [
                    "with an expression revealing disappointment",
                    "showing signs of frustration at the challenge",
                    "with concern evident during a difficult moment",
                    "visibly processing the emotional weight of competition"
                ]
            elif emotion == 'angry' or emotion == 'anger':
                emotion_options = [
                    "with fierce intensity in their eyes",
                    "showing the powerful determination of an elite competitor",
                    "with an expression reflecting competitive fire",
                    "channeling powerful emotion into performance"
                ]
            elif emotion == 'surprise':
                emotion_options = [
                    "with astonishment at the unfolding situation",
                    "showing surprise at the unexpected development",
                    "with a notable expression of shock",
                    "displaying visible reaction to the surprising turn of events"
                ]
            elif emotion == 'determination':
                emotion_options = [
                    "with resolute determination etched on their face",
                    "showing unyielding commitment in their expression",
                    "with the steadfast focus of an elite athlete",
                    "displaying the mental fortitude required at this level"
                ]
            else:
                emotion_options = [
                    "with clearly visible emotion",
                    "displaying the powerful feelings of competition",
                    "with an expressive reaction to the moment",
                    "showing the emotional intensity of athletics"
                ]

            emotion_phrases.append(random.choice(emotion_options))

    # ----------------- 8. CLOSING PHRASES -----------------
    if action_level > 0.7:
        closing_options = [
            "The image perfectly captures this remarkable athletic moment.",
            "This photograph brilliantly conveys the intensity and skill of competitive sports.",
            "This powerful image freezes a memorable moment of athletic excellence.",
            "The photograph vividly showcases the peak of sporting achievement."
        ]
    elif action_level > 0.4:
        closing_options = [
            "The image provides an authentic glimpse into the nature of this sport.",
            "This photograph effectively highlights the technical aspects of athletic performance.",
            "The image captures the essence of this sporting discipline.",
            "This compelling viewpoint reveals the nuanced dynamics of the competition."
        ]
    else:
        closing_options = [
            "The image reveals a rarely seen calm moment in this intense sport.",
            "This photograph offers a different perspective on the athletic world.",
            "The image captures a moment of preparation that precedes athletic action.",
            "This thoughtful composition shows the mental aspect of sports competition."
        ]

    closing_phrases.append(random.choice(closing_options))

    # ----------------- 9. COMBINE ALL COMPONENTS -----------------
    all_parts = []

    # Add intro
    if intro_phrases:
        all_parts.append(intro_phrases[0])

    # Add subject
    if subject_phrases:
        all_parts.append(subject_phrases[0])

    # Add action
    if action_phrases:
        all_parts.append(action_phrases[0])

    # Add details
    if detail_phrases:
        all_parts.append(detail_phrases[0])

    # Add emotion
    if emotion_phrases:
        all_parts.append(emotion_phrases[0])

    # Combine main parts
    main_caption = ' '.join(all_parts)

    # Add closing
    if closing_phrases:
        caption = main_caption + '. ' + closing_phrases[0]
    else:
        caption = main_caption + '.'

    # Ensure proper capitalization and punctuation
    caption = caption.strip()
    if not caption.endswith('.'):
        caption += '.'

    # Fix spacing issues
    caption = caption.replace('  ', ' ')
    caption = caption.replace(' ,', ',')
    caption = caption.replace(' .', '.')

    return caption




def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sports Image Analysis')
    parser.add_argument('--image', type=str, help='Path to image file')
    args = parser.parse_args()

    # Check dependencies first
    print("Checking and installing dependencies...")
    check_dependencies()

    # Either use the provided image path or ask for one
    if args.image:
        image_path = args.image
    else:
        image_path = input("Please enter the path to the sports image: ")

    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found")
        return

    # Analyze image
    print(f"Analyzing image: {image_path}")
    analysis = analyze_sports_image(image_path)
    print("Analysis complete. Results saved to sports_analysis_results.png and analysis_results.txt")
    return analysis


if __name__ == "__main__":
    main()
