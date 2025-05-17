import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import torch, cv2, numpy as np, matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from scipy import ndimage
import time
import argparse
import sys
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# Sửa lỗi matplotlib warning khi chạy trong thread khác
import matplotlib
matplotlib.use('Agg')  # Sử dụng Agg backend thay vì interactive backend

# Tiếp tục với các import khác...

# Function to install required packages if not already installed
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
        'scipy': 'scipy',
        'PIL': 'pillow'
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

        return midas, yolo, device

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

    # Extract detections
    boxes = []
    classes = []
    scores = []
    sports_classes = ['person', 'sports ball', 'tennis racket', 'baseball bat', 'baseball glove',
                      'skateboard', 'surfboard', 'tennis ball', 'bottle', 'wine glass', 'cup',
                      'frisbee', 'skis', 'snowboard', 'kite']

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
    """Phân tích độ sắc nét của từng đối tượng trong ảnh"""
    # Chuyển sang ảnh grayscale nếu đầu vào là RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    sharpness_scores = []
    sharpness_details = []

    for box in boxes:
        x1, y1, x2, y2 = box
        # Giới hạn trong phạm vi ảnh
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)

        # Kiểm tra nếu box hợp lệ
        if x1 >= x2 or y1 >= y2:
            sharpness_scores.append(0)
            sharpness_details.append({
                'laplacian_var': 0,
                'sobel_mean': 0,
                'combined_score': 0
            })
            continue

        # Trích xuất ROI
        roi = gray[y1:y2, x1:x2]

        # Áp dụng bộ lọc Laplacian để phát hiện cạnh
        lap = cv2.Laplacian(roi, cv2.CV_64F)
        lap_var = lap.var()

        # Tính gradient sử dụng Sobel
        sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        sobel_mean = np.mean(sobel_magnitude)

        # Kết hợp các số đo (trọng số có thể điều chỉnh)
        sharpness = lap_var * 0.6 + sobel_mean * 0.4
        sharpness_scores.append(sharpness)

        sharpness_details.append({
            'laplacian_var': lap_var,
            'sobel_mean': sobel_mean,
            'combined_score': sharpness
        })

    # Chuẩn hóa điểm
    if sharpness_scores:
        max_score = max(sharpness_scores) if max(sharpness_scores) > 0 else 1
        sharpness_scores = [score / max_score for score in sharpness_scores]

    return sharpness_scores, sharpness_details


def analyze_sports_scene(detections, depth_map, img_data):
    """Analyze the sports scene based on detected objects and depth"""
    height, width = depth_map.shape[:2]

    # Analyze player distribution
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

    # Initialize player_dispersion with default value
    player_dispersion = 0

    # Calculate player dispersion (if multiple players)
    if len(player_positions) > 1:
        # Calculate average pairwise distance
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

    # MỚI: Phân tích độ sắc nét của các đối tượng
    image = img_data['resized_array']
    sharpness_scores, sharpness_details = analyze_object_sharpness(image, detections['boxes'])
    print(f"Sharpness scores: {[f'{score:.2f}' for score in sharpness_scores]}")

    # Identify key subjects (based on size and position)
    key_subjects = []
    for i, box in enumerate(detections['boxes']):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        area_ratio = area / (img_data['resized_array'].shape[1] * img_data['resized_array'].shape[0])

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

        subject_info = {
            'class': detections['classes'][i],
            'box': box,
            'area_ratio': area_ratio,
            'depth': obj_depth,
            'sharpness': sharpness,  # MỚI: Thêm độ sắc nét
            'sharpness_details': sharpness_details[i] if i < len(sharpness_details) else None,
            # MỚI: Chi tiết độ sắc nét
            'position': ((x1 + x2) / 2 / img_data['resized_array'].shape[1],
                         (y1 + y2) / 2 / img_data['resized_array'].shape[0])
        }

        # MỚI: Kết hợp vị trí, kích thước, độ sâu và độ sắc nét
        center_dist = np.sqrt((subject_info['position'][0] - 0.5) ** 2 +
                              (subject_info['position'][1] - 0.5) ** 2)

        # Tính toán vị trí và kích thước (40%)
        position_size_score = area_ratio * (1 - min(1.0, center_dist))

        # Độ sắc nét (40%)
        sharpness_score = sharpness

        # Đối tượng gần hơn có điểm cao hơn (20%)
        depth_score = 1 - obj_depth

        # MỚI: Tổng hợp điểm số với trọng số
        subject_info['prominence'] = position_size_score * 0.4 + sharpness_score * 0.4 + depth_score * 0.2

        key_subjects.append(subject_info)

    # Sort by prominence
    key_subjects.sort(key=lambda x: x['prominence'], reverse=True)

    return {
        'player_count': detections['athletes'],
        'player_positions': player_positions,
        'player_dispersion': player_dispersion,
        'key_subjects': key_subjects[:5] if key_subjects else [],
        'sharpness_scores': sharpness_scores  # MỚI: Lưu điểm số sắc nét
    }


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
    """Simplified face verification especially for sports images"""
    try:
        # 1. Check minimum dimensions
        if face_img.shape[0] < 5 or face_img.shape[1] < 5:
            return False, "Face too small (under 5px)"

        # 2. Just check very basic aspect ratio
        h, w = face_img.shape[:2]
        aspect_ratio = h / w
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:  # Very generous
            return False, "Extremely unusual face ratio"

        # 3. Check it's not a monochrome region
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        if np.std(gray) < 10:  # Only reject nearly monochrome regions
            return False, "Image region too monotonous"

        # NEW: Skip other verification steps entirely
        return True, "Face accepted in sports context"

    except Exception as e:
        return False, f"Verification error: {str(e)}"


def analyze_sports_composition(detections, analysis, img_data):
    """Analyze the composition with sports-specific context"""

    # Basic composition from existing analysis
    composition = analysis["composition_analysis"] if "composition_analysis" in analysis else {}

    # Sports specific enhancements
    result = {
        'sport_type': 'Unknown',
        'framing_quality': 'Unknown',
        'recommended_crop': None,
        'action_focus': 'Unknown'
    }

    # Try to determine sport type
    sport_equipment = {
        'tennis racket': 'Tennis',
        'tennis ball': 'Tennis',
        'sports ball': 'Ball Sport',
        'baseball bat': 'Baseball',
        'baseball glove': 'Baseball',
        'skateboard': 'Skateboarding',
        'surfboard': 'Surfing',
        'frisbee': 'Frisbee',
        'skis': 'Skiing',
        'snowboard': 'Snowboarding'
    }

    for cls in detections['classes']:
        if cls in sport_equipment:
            result['sport_type'] = sport_equipment[cls]
            break

    # Evaluate framing quality for sports action
    if "key_subjects" in analysis and analysis['key_subjects']:
        subject_positions = [subject['position'] for subject in analysis['key_subjects']]

        # Check if key subjects are well placed (rule of thirds or centered)
        well_placed_count = 0
        for pos in subject_positions:
            # Check rule of thirds points
            thirds_points = [
                (1 / 3, 1 / 3), (2 / 3, 1 / 3),
                (1 / 3, 2 / 3), (2 / 3, 2 / 3)
            ]

            center_point = (0.5, 0.5)

            # Check if close to rule of thirds points or center
            for third in thirds_points:
                dist = np.sqrt((pos[0] - third[0]) ** 2 + (pos[1] - third[1]) ** 2)
                if dist < 0.1:  # 10% of image width/height
                    well_placed_count += 1
                    break

            # Check if centered
            dist_to_center = np.sqrt((pos[0] - center_point[0]) ** 2 + (pos[1] - center_point[1]) ** 2)
            if dist_to_center < 0.1:
                well_placed_count += 1

        if well_placed_count / len(subject_positions) > 0.7:
            result['framing_quality'] = 'Excellent'
        elif well_placed_count / len(subject_positions) > 0.4:
            result['framing_quality'] = 'Good'
        else:
            result['framing_quality'] = 'Could be improved'

    # Recommend crop if needed
    if "key_subjects" in analysis and analysis['key_subjects']:
        main_subject = analysis['key_subjects'][0]
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
    """Phân tích biểu cảm khuôn mặt nâng cao với OpenCV và HSEmotion"""
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
        img_area = image.shape[0] * image.shape[1]

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

        # 2. EXTRACT FACE REGION FROM MAIN SUBJECT
        x1, y1, x2, y2 = main_subject['box']

        # Lưu ảnh đối tượng chính để debug
        subject_img = image[max(0, y1):min(y2, image.shape[0]),
                      max(0, x1):min(x2, image.shape[1])]
        subject_path = f"{debug_dir}/main_subject.jpg"
        cv2.imwrite(subject_path, cv2.cvtColor(subject_img, cv2.COLOR_RGB2BGR))

        # 3. PHÁT HIỆN KHUÔN MẶT VỚI OPENCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

        # Chuyển sang grayscale
        gray_subject = cv2.cvtColor(subject_img, cv2.COLOR_RGB2GRAY)

        # Cải thiện tỷ lệ phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(gray_subject, 1.05, 3, minSize=(20, 20))

        # Nếu không tìm thấy, thử với khuôn mặt nghiêng
        if len(faces) == 0:
            faces = profile_cascade.detectMultiScale(gray_subject, 1.05, 2, minSize=(20, 20))

        # Nếu vẫn không tìm thấy, thử với phần trên của đối tượng
        if len(faces) == 0:
            # Tạo vùng tìm kiếm mới - phần trên cơ thể (40% phần trên)
            h, w = subject_img.shape[:2]
            head_height = int(h * 0.4)
            head_region = subject_img[0:head_height, 0:w]

            if head_region.size > 0:
                # Lưu để debug
                head_path = f"{debug_dir}/head_region.jpg"
                cv2.imwrite(head_path, cv2.cvtColor(head_region, cv2.COLOR_RGB2BGR))

                # Thử phát hiện lại trong vùng đầu
                gray_head = cv2.cvtColor(head_region, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray_head, 1.03, 2, minSize=(15, 15))

                # Nếu vẫn không tìm thấy, thử với vùng đầu nghiêng
                if len(faces) == 0:
                    faces = profile_cascade.detectMultiScale(gray_head, 1.03, 2, minSize=(15, 15))

        # Nếu vẫn không tìm thấy, sử dụng toàn bộ vùng đầu làm khuôn mặt
        if len(faces) == 0:
            h, w = subject_img.shape[:2]
            # Ước lượng khuôn mặt chiếm 35% trên cùng của đối tượng
            est_face_h = int(h * 0.35)
            est_face_w = int(est_face_h * 0.8)  # Aspect ratio ~0.8
            est_face_x = (w - est_face_w) // 2  # Căn giữa
            est_face_y = 0

            faces = [(est_face_x, est_face_y, est_face_w, est_face_h)]
            print("Using estimated face region from upper part of subject")

        # Kiểm tra nếu tìm thấy khuôn mặt
        face_found = len(faces) > 0
        face_img = None
        fx, fy, fw, fh = 0, 0, 0, 0

        if face_found:
            print(f"Found {len(faces)} faces")

            # Lấy khuôn mặt lớn nhất
            best_face = max(faces, key=lambda f: f[2] * f[3])
            hx, hy, hw, hh = best_face

            # Điều chỉnh tọa độ về khung hình gốc
            fx, fy, fw, fh = hx, hy, hw, hh

            # Trích xuất khuôn mặt
            # Thêm padding xung quanh khuôn mặt để cải thiện kết quả
            padding = max(10, int(hw * 0.1))  # Padding tỷ lệ với kích thước khuôn mặt
            face_x1 = max(0, hx - padding)
            face_y1 = max(0, hy - padding)
            face_x2 = min(subject_img.shape[1], hx + hw + padding)
            face_y2 = min(subject_img.shape[0], hy + hh + padding)

            face_img = subject_img[face_y1:face_y2, face_x1:face_x2]

            # Lưu khuôn mặt để debug
            best_face_path = f"{debug_dir}/best_face.jpg"
            cv2.imwrite(best_face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            print(f"Face dimensions: {face_img.shape[1]}x{face_img.shape[0]}")

            # Điều chỉnh tọa độ khuôn mặt về không gian hình ảnh gốc
            fx += x1
            fy += y1

        if not face_found or face_img is None or face_img.size == 0:
            print("No face detected after trying multiple methods")
            expression_results['debug_info']['reason'] = "No face detected in the subject"
            return expression_results

        # PHÂN TÍCH BIỂU CẢM SỬ DỤNG MODEL CÓ SẴN TRONG DNN MODELS
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

            # GIẢI PHÁP CUỐI CÙNG: PHÂN TÍCH LBP + HOG FEATURES
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
                    'contempt': 0.1
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

                # Phần phân tích cơ bản (giữ nguyên)
            # Phân tích đơn giản dựa trên độ tương phản và độ sáng
            gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray) / 255.0
            contrast = np.std(gray) / 128.0

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
            top_contrast = np.std(top) / 128.0
            middle_contrast = np.std(middle) / 128.0
            bottom_contrast = np.std(bottom) / 128.0

            # Khởi tạo điểm số cảm xúc
            emotion_scores_dict = {
                'neutral': 0.10,
                'happy': 0.10,
                'surprise': 0.10,
                'sad': 0.10,
                'angry': 0.10,
                'fear': 0.10,
                'disgust': 0.10
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
        sport_type = "Unknown"
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

        # 8. HIỂN THỊ KẾT QUẢ
        def visualize_emotion_results(face, emotion_name, intensity, emotions_dict):
            """Hiển thị kết quả phân tích cảm xúc"""
            # Khởi tạo hình ảnh hiển thị
            result_img = face.copy()

            # Vẽ nhãn cảm xúc
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result_img, f"{emotion_name} ({intensity:.2f})", (10, 25), font, 0.7, (0, 255, 0), 2)

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
                'determination': (0.9, 0.3, 0.0),  # Deep Orange - màu mới cho quyết tâm
                'effort': (0.7, 0.2, 0.4),  # Cranberry - màu mới cho nỗ lực
                'unknown': (0.7, 0.7, 0.7)  # Light Gray
            }

            # Vẽ thanh cảm xúc
            y_offset = 40
            bar_height = 15
            max_width = result_img.shape[1] - 20

            for emotion, score in sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
                # Tính toán màu
                color = emotion_colors.get(emotion, (0.5, 0.5, 0.5))  # Default: gray
                color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))

                # Vẽ thanh
                bar_width = int(score * max_width)
                cv2.rectangle(result_img, (10, y_offset), (10 + bar_width, y_offset + bar_height), color_bgr, -1)

                # Vẽ nhãn
                cv2.putText(result_img, f"{emotion}: {score:.2f}",
                            (10, y_offset - 5), font, 0.5, color_bgr, 1)

                # Cập nhật offset cho thanh tiếp theo
                y_offset += bar_height + 15

            # Lưu kết quả
            result_path = f"{debug_dir}/emotion_result.jpg"
            cv2.imwrite(result_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            print(f"Emotion visualization saved to {result_path}")

        # Tạo hình ảnh hiển thị kết quả cảm xúc
        visualize_emotion_results(face_img, dominant_emotion, emotion_intensity, contextual_emotions)

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
                             facial_analysis=None):
    """Create sports-specific visualization with enhanced main subject highlighting and emotion analysis"""
    img = np.array(img_data['resized']).copy()
    height, width = img.shape[:2]

    # Tìm đối tượng chính (người) từ key_subjects
    main_person = None
    main_person_idx = -1

    if "key_subjects" in sports_analysis:
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

                # Thêm tên nhãn nổi bật hơn
                cv2.putText(main_obj_viz, "MAIN SUBJECT", (x1, y1 - 25),
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
        cv2.putText(det_viz, f"{label} {conf:.2f} S:{sharpness:.2f}", (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    # Tạo hiệu ứng làm mờ hình nền và làm nổi bật đối tượng chính
    if main_person is not None:
        # Blur background
        blurred_bg = cv2.GaussianBlur(img, (25, 25), 0)

        # Tạo mask cho vùng người chính
        x1, y1, x2, y2 = main_person['box']
        person_mask = np.zeros((height, width), dtype=np.uint8)
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
        cv2.rectangle(main_highlight, (x1, y1), (x2, y2), color, 3)
        cv2.putText(main_highlight, "MAIN SUBJECT", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
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

            # Hiển thị điểm sắc nét và chỉ số prominence
            label_text = f"P:{subject['prominence']:.2f} S:{subject.get('sharpness', 0):.2f}"
            if is_main_subject:
                label_text = "MAIN: " + label_text

            cv2.putText(comp_viz, label_text,
                        (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Tạo heatmap độ sắc nét
    sharpness_viz = img.copy()

    # Sử dụng colormap
    from matplotlib import cm
    jet_colormap = cm.get_cmap('jet')

    for i, box in enumerate(detections['boxes']):
        if i < len(sharpness_scores):
            x1, y1, x2, y2 = box
            sharpness = sharpness_scores[i]

            # Màu dựa trên độ sắc nét
            color_rgba = jet_colormap(sharpness)
            color_rgb = tuple(int(255 * c) for c in color_rgba[:3])
            color = (color_rgb[2], color_rgb[1], color_rgb[0])  # BGR for OpenCV

            # Vẽ heatmap với opacity dựa trên độ sắc nét
            overlay = sharpness_viz.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # Filled rectangle

            # Thêm overlay với alpha blending
            alpha = 0.4 + 0.3 * sharpness  # More opaque for sharper objects
            cv2.addWeighted(overlay, alpha, sharpness_viz, 1 - alpha, 0, sharpness_viz)

            # Thêm border và nhãn
            cv2.rectangle(sharpness_viz, (x1, y1), (x2, y2), color, 3)
            cv2.putText(sharpness_viz, f"Sharp: {sharpness:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Hiển thị biểu cảm khuôn mặt
    def visualize_emotion_results(face_img, emotion_analysis):
        """Tạo hiển thị chuyên nghiệp cho phân tích biểu cảm khuôn mặt"""
        if face_img is None or emotion_analysis is None or not emotion_analysis.get('has_faces', False):
            return None

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

    # PHÂN TÍCH VÀ HIỂN THỊ BIỂU CẢM KHUÔN MẶT CẢI TIẾN
    face_emotion_viz = None
    if facial_analysis and facial_analysis.get('has_faces', False):
        # Tìm ảnh khuôn mặt
        face_img = None
        try:
            if 'face_path' in facial_analysis:
                face_img = cv2.imread(facial_analysis['face_path'])
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            else:
                face_img = cv2.imread("face_debug/best_face.jpg")
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Could not load face image: {str(e)}")
            face_img = None

        if face_img is not None:
            # Tạo hiển thị biểu cảm nâng cao
            face_emotion_viz = visualize_emotion_results(face_img, facial_analysis)

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

    # Hiển thị với bố cục nâng cao
    fig = plt.figure(figsize=(18, 12))

    # Định nghĩa lưới hiển thị với kích thước khác nhau
    grid = plt.GridSpec(3, 4, hspace=0.3, wspace=0.3)

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

    # Face analysis nâng cao
    if face_emotion_viz is not None:
        ax_face = fig.add_subplot(grid[2, 2:4])
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
        # MỚI: Hiển thị thông báo rõ ràng khi không phát hiện được khuôn mặt
        ax_info = fig.add_subplot(grid[2, 2:4])

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

    #plt.tight_layout()
    plt.savefig("sports_analysis_results.png", dpi=150)
    #plt.show()

    # Print detailed analysis
    print("\n==== SPORTS IMAGE ANALYSIS ====")
    print(
        f"Detected {detections['athletes']} athletes and {len(detections['classes']) - detections['athletes']} other objects")

    if "sport_type" in composition_analysis:
        print(f"\nSport type: {composition_analysis['sport_type']}")

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

        f.write("\nComposition Analysis:\n")
        f.write(f"- Framing quality: {composition_analysis['framing_quality']}\n")

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


def analyze_sports_image(file_path):
    """Main function to analyze sports images"""
    t_start = time.time()

    # Load models
    print("Loading models...")
    midas, yolo, device = load_models()
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
    sports_analysis = analyze_sports_scene(detections, depth_map, img_data)

    # Step 4: Analyze action quality
    print("Analyzing action quality...")
    action_analysis = analyze_action_quality(detections, img_data)

    # Step 5: Sports composition analysis
    print("Analyzing composition...")
    composition_analysis = analyze_sports_composition(detections, {'sports_analysis': sports_analysis}, img_data)

    # Step 6: Facial expression analysis với phiên bản nâng cao
    print("Starting advanced facial expression analysis...")
    try:
        facial_analysis = analyze_facial_expression_advanced(
            detections,
            img_data,
            depth_map=depth_map,
            sports_analysis={'sports_analysis': sports_analysis, 'composition_analysis': composition_analysis}
        )
        print("Advanced facial expression analysis successful:", facial_analysis.get('has_faces', False))
    except Exception as e:
        print(f"Error in facial expression analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        facial_analysis = {'has_faces': False, 'error': str(e)}

    # Step 7: Visualize results với hiển thị biểu cảm cải tiến
    print("Visualizing results...")
    visualize_sports_results(img_data, detections, depth_map,
                             sports_analysis, action_analysis, composition_analysis,
                             facial_analysis)

    t_end = time.time()
    print(f"\nAnalysis completed in {t_end - t_start:.2f} seconds")

    return {
        'detections': detections,
        'sports_analysis': sports_analysis,
        'action_analysis': action_analysis,
        'composition_analysis': composition_analysis,
        'facial_analysis': facial_analysis
    }


def generate_sports_caption(analysis_result):
    """
    Generates a descriptive caption for sports images based on analysis results.

    Args:
        analysis_result: Dictionary containing sports image analysis data

    Returns:
        str: A well-formatted caption describing the sports image
    """
    # Extract key information from analysis results
    detections = analysis_result.get('detections', {})
    sports_analysis = analysis_result.get('sports_analysis', {})
    action_analysis = analysis_result.get('action_analysis', {})
    composition_analysis = analysis_result.get('composition_analysis', {})
    facial_analysis = analysis_result.get('facial_analysis', {})

    # Initialize caption parts
    parts = []

    # 1. Sport type and setting
    sport_type = composition_analysis.get('sport_type', 'Unknown')
    if sport_type != 'Unknown':
        parts.append(f"A {sport_type} action shot")
    else:
        parts.append("A sports action shot")

    # 2. Athletes information
    athlete_count = detections.get('athletes', 0)
    if athlete_count > 0:
        if athlete_count == 1:
            parts.append(f"featuring a solo athlete")
        else:
            parts.append(f"featuring {athlete_count} athletes")

    # 3. Action quality
    action_level = action_analysis.get('action_level', 0)
    action_quality = action_analysis.get('action_quality', '')

    if action_quality:
        if action_quality == 'High':
            action_phrase = "in an intense moment of action"
        elif action_quality == 'Medium':
            action_phrase = "during active play"
        else:
            action_phrase = "in a calm moment"
        parts.append(action_phrase)

    # 4. Equipment
    equipment = action_analysis.get('equipment_types', [])
    if equipment:
        equipment_str = ', '.join(equipment)
        parts.append(f"with {equipment_str}")

    # 5. Composition quality
    framing = composition_analysis.get('framing_quality', '')
    if framing:
        if framing == 'Excellent':
            parts.append("perfectly framed")
        elif framing == 'Good':
            parts.append("well composed")
        elif framing == 'Could be improved':
            # Omit negative composition feedback from captions
            pass

    # 6. Main subject highlight
    key_subjects = sports_analysis.get('key_subjects', [])
    if key_subjects:
        main_subject = key_subjects[0]
        if main_subject.get('class') == 'person' and main_subject.get('prominence', 0) > 0.5:
            parts.append("highlighting the main athlete")

    # 7. Emotional aspect (if face detected)
    if facial_analysis and facial_analysis.get('has_faces', False):
        emotion = facial_analysis.get('dominant_emotion', '')
        intensity = facial_analysis.get('emotion_intensity', 0)

        if emotion and intensity > 0.4:
            if emotion.lower() == 'happy':
                emotion_phrase = "displaying joy and excitement"
            elif emotion.lower() == 'neutral':
                emotion_phrase = "with focused determination"
            elif emotion.lower() == 'sad':
                emotion_phrase = "showing signs of disappointment"
            elif emotion.lower() == 'angry':
                emotion_phrase = "with intense competitive spirit"
            elif emotion.lower() == 'surprise':
                emotion_phrase = "with a look of surprise"
            else:
                emotion_phrase = f"expressing {emotion}"

            parts.append(emotion_phrase)

    # Combine all parts into a coherent caption
    caption = ' '.join(parts)

    # Add final touch
    if action_level > 0.7:
        caption += ". The image captures the peak moment of athletic performance."
    elif 0.4 <= action_level <= 0.7:
        caption += ". The image effectively conveys the dynamic nature of the sport."
    else:
        caption += ". The image provides an insightful look at the sport."

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