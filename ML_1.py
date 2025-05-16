import torch, os, cv2, numpy as np, matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from scipy import ndimage
import time
import argparse
import sys
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap


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
    """Phân tích biểu cảm khuôn mặt nâng cao với kiểm tra hướng mặt"""
    try:
        print("Starting advanced facial expression analysis...")

        import cv2
        import numpy as np
        import os

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

                    # Trọng số kết hợp (diện tích + vị trí trung tâm + độ sâu)
                    center_weight = area * (1 - min(1.0, dist_to_center))

                    # Thêm độ sâu nếu có
                    if depth_map is not None:
                        # Điều chỉnh tọa độ cho depth_map
                        depth_x1 = max(0, min(int(x1 * depth_map.shape[1] / width), depth_map.shape[1] - 1))
                        depth_y1 = max(0, min(int(y1 * depth_map.shape[0] / height), depth_map.shape[0] - 1))
                        depth_x2 = max(0, min(int(x2 * depth_map.shape[1] / width), depth_map.shape[1] - 1))
                        depth_y2 = max(0, min(int(y2 * depth_map.shape[0] / height), depth_map.shape[0] - 1))

                        # Tính độ sâu trung bình (giá trị nhỏ hơn = gần camera hơn)
                        depth_region = depth_map[depth_y1:depth_y2, depth_x1:depth_x2]
                        avg_depth = np.mean(depth_region) if depth_region.size > 0 else 1.0

                        # Đối tượng gần có weight cao hơn
                        depth_weight = 1 - avg_depth
                        center_weight *= (1 + depth_weight)

                    # Lưu giá trị cao nhất
                    if center_weight > max_center_weight:
                        max_center_weight = center_weight
                        main_subject = {
                            'box': box,
                            'class': cls,
                            'prominence': center_weight,
                            'depth': avg_depth if depth_map is not None else None
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

        # Cải thiện khu vực ước tính khuôn mặt cho vận động viên
        # Người chạy thường có đầu cao hơn so với phần thân
        # Sử dụng phần trên lớn hơn (33% thay vì 25%)
        head_height = int((y2 - y1) * 0.33)  # Tăng từ 25% lên 33%

        # Mở rộng vùng tìm kiếm khuôn mặt hơn một chút
        face_y1 = max(0, y1 - int(head_height * 0.1))  # Mở rộng lên trên thêm chút
        face_y2 = min(y1 + head_height, image.shape[0])
        face_x1 = max(0, x1)
        face_x2 = min(x2, image.shape[1])

        # Trích xuất khu vực khuôn mặt ước tính
        face_region = image[face_y1:face_y2, face_x1:face_x2]
        face_path = f"{debug_dir}/face_region_estimate.jpg"
        cv2.imwrite(face_path, cv2.cvtColor(face_region, cv2.COLOR_RGB2BGR))

        # 3. TÌM KHUÔN MẶT CHÍNH XÁC HƠN VỚI MTCNN
        try:
            from mtcnn import MTCNN
            detector = MTCNN(min_face_size=20)  # Đặt kích thước khuôn mặt tối thiểu nhỏ hơn
        except ImportError:
            print("Installing MTCNN...")
            os.system("pip install mtcnn")
            from mtcnn import MTCNN
            detector = MTCNN(min_face_size=20)

        # Giảm ngưỡng confidence cho ảnh thể thao - chuyển động
        confidence_threshold = 0.7  # Giảm từ 0.9 xuống 0.7

        # A. Tìm trong vùng đầu ước tính
        faces = detector.detect_faces(face_region)
        face_found = False
        best_face = None

        if faces:
            print(f"Found {len(faces)} faces in estimated head region")

            # Lấy khuôn mặt lớn nhất và đáng tin cậy nhất
            confident_faces = [face for face in faces if face['confidence'] > confidence_threshold]
            if confident_faces:
                best_face = max(confident_faces, key=lambda face: face['box'][2] * face['box'][3])
                face_found = True
                print(f"Found confident face with confidence: {best_face['confidence']:.2f}")

                # Điều chỉnh tọa độ về khung hình gốc
                fx, fy, fw, fh = best_face['box']
                fx += face_x1
                fy += face_y1

                print(f"Face dimensions: {fw}x{fh}")
            else:
                print("No confident face found in head region")

        # B. Nếu không tìm thấy trong vùng đầu, thử với toàn bộ phần trên của cơ thể
        if not face_found:
            # Tạo vùng tìm kiếm mới - phần trên cơ thể (50% phần trên)
            upper_body_y1 = max(0, y1)
            upper_body_y2 = min(y1 + int((y2 - y1) * 0.5), image.shape[0])
            upper_body_x1 = max(0, x1)
            upper_body_x2 = min(x2, image.shape[1])

            upper_body = image[upper_body_y1:upper_body_y2, upper_body_x1:upper_body_x2]
            upper_body_path = f"{debug_dir}/upper_body.jpg"
            cv2.imwrite(upper_body_path, cv2.cvtColor(upper_body, cv2.COLOR_RGB2BGR))

            # Tìm khuôn mặt trong phần trên
            faces = detector.detect_faces(upper_body)
            if faces:
                print(f"Found {len(faces)} faces in upper body region")

                # Lấy khuôn mặt lớn nhất và đáng tin cậy nhất
                confident_faces = [face for face in faces if face['confidence'] > confidence_threshold]
                if confident_faces:
                    best_face = max(confident_faces, key=lambda face: face['box'][2] * face['box'][3])
                    face_found = True
                    print(f"Found confident face with confidence: {best_face['confidence']:.2f}")

                    # Điều chỉnh tọa độ về khung hình gốc
                    fx, fy, fw, fh = best_face['box']
                    fx += upper_body_x1
                    fy += upper_body_y1

                    print(f"Face dimensions: {fw}x{fh}")

        # C. Nếu vẫn không tìm thấy, thử với toàn bộ đối tượng
        if not face_found:
            faces = detector.detect_faces(subject_img)
            if faces:
                print(f"Found {len(faces)} faces in full main subject")

                # Lọc ra các khuôn mặt đáng tin cậy
                confident_faces = [face for face in faces if face['confidence'] > confidence_threshold]
                if confident_faces:
                    # Ưu tiên khuôn mặt ở phía trên và lớn
                    def face_score(face):
                        fx, fy, fw, fh = face['box']
                        size_score = fw * fh  # Kích thước lớn tốt hơn
                        position_score = -fy * 2  # Vị trí cao hơn tốt hơn
                        return size_score + position_score

                    best_face = max(confident_faces, key=face_score)
                    face_found = True

                    # Điều chỉnh tọa độ về khung hình gốc
                    fx, fy, fw, fh = best_face['box']
                    fx += x1
                    fy += y1

                    print(f"Face dimensions: {fw}x{fh}")
                else:
                    print("No confident face found in subject")

        # D. Nếu vẫn không tìm thấy, sử dụng detector khác nếu có thể
        if not face_found:
            try:
                import cv2
                # Thử với OpenCV's Haar Cascade - xử lý tốt hơn với góc nhìn khó
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

                # Chuyển sang grayscale
                gray_subject = cv2.cvtColor(subject_img, cv2.COLOR_RGB2GRAY)

                # Phát hiện khuôn mặt nhìn thẳng
                faces = face_cascade.detectMultiScale(gray_subject, 1.1, 4)

                # Nếu không tìm thấy, thử với khuôn mặt nghiêng
                if len(faces) == 0:
                    faces = profile_cascade.detectMultiScale(gray_subject, 1.1, 4)

                # Nếu phát hiện được
                if len(faces) > 0:
                    print(f"Found {len(faces)} faces with OpenCV Haar Cascade")

                    # Lấy khuôn mặt lớn nhất
                    best_haar_face = max(faces, key=lambda f: f[2] * f[3])
                    hx, hy, hw, hh = best_haar_face

                    # Chuyển đổi sang định dạng giống MTCNN
                    best_face = {
                        'box': (hx, hy, hw, hh),
                        'confidence': 0.8,  # Giả định confidence
                        'keypoints': {}
                    }

                    # Điều chỉnh tọa độ về khung hình gốc
                    fx, fy, fw, fh = hx, hy, hw, hh
                    fx += x1
                    fy += y1

                    face_found = True
                    print(f"Face found with Haar Cascade: {fw}x{fh}")
            except Exception as e:
                print(f"Error using OpenCV Haar Cascade: {str(e)}")

        # Kiểm tra xem có tìm thấy khuôn mặt không
        if not face_found:
            print("No face detected after trying multiple methods")
            expression_results['debug_info']['reason'] = "No face detected in the subject"
            return expression_results

        # 4. KIỂM TRA HƯỚNG KHUÔN MẶT VÀ ĐẶC ĐIỂM KHUÔN MẶT
        def verify_face_orientation(face_img, keypoints=None):
            """Kiểm tra xem khuôn mặt có hướng về phía camera không"""
            # Nếu không có keypoints, thử nhận diện với dlib
            try:
                # Chuyển đổi ảnh sang grayscale
                gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)

                # Kiểm tra keypoints từ MTCNN
                if keypoints and all(k in keypoints for k in ['left_eye', 'right_eye', 'nose']):
                    print("Verified face using MTCNN keypoints")
                    return True

                # Kiểm tra bằng phân bố gradient và các đặc trưng khuôn mặt
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

                # Chia ảnh thành grid 3x3
                h, w = gray.shape
                cell_h, cell_w = h // 3, w // 3

                # Kiểm tra đặc trưng khuôn mặt phân bố
                center_cell = gradient_mag[cell_h:2 * cell_h, cell_w:2 * cell_w]
                center_mean = np.mean(center_cell)

                # Vùng mắt (thường là phần trên của khuôn mặt)
                eye_region = gradient_mag[:cell_h, :]
                eye_mean = np.mean(eye_region)

                # Vùng miệng (thường là phần dưới của khuôn mặt)
                mouth_region = gradient_mag[2 * cell_h:, :]
                mouth_mean = np.mean(mouth_region)

                # Khuôn mặt thường có đặc trưng tại mắt và miệng (gradient cao)
                if eye_mean > 30 and mouth_mean > 20 and center_mean > 15:
                    print(
                        f"Face features verified: eye={eye_mean:.1f}, mouth={mouth_mean:.1f}, center={center_mean:.1f}")
                    return True
                else:
                    print(
                        f"Insufficient face features: eye={eye_mean:.1f}, mouth={mouth_mean:.1f}, center={center_mean:.1f}")

                    # Fallback: Kiểm tra phân phối grayscale
                    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    hist_norm = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                    # Tính entropy của histogram (khuôn mặt thường có entropy cao)
                    hist_flat = hist_norm.flatten() + 1e-7  # Tránh log(0)
                    entropy = -np.sum(hist_flat * np.log2(hist_flat))

                    # Tính độ tương phản (khuôn mặt thường có độ tương phản cao)
                    contrast = np.std(gray)

                    # Kiểm tra phân phối màu da
                    # (Trong không gian HSV, màu da con người có hue khoảng 0-50)
                    hsv = cv2.cvtColor(face_img, cv2.COLOR_RGB2HSV)
                    h, s, v = cv2.split(hsv)

                    # Tính phần trăm pixel trong vùng màu da
                    skin_mask = ((h >= 0) & (h <= 50) & (s >= 25) & (s <= 230)) | (h < 20)
                    skin_percent = np.sum(skin_mask) / (face_img.shape[0] * face_img.shape[1])

                    # In thông tin điểm số
                    print(
                        f"Face verification: entropy={entropy:.2f}, contrast={contrast:.2f}, skin%={skin_percent:.2f}")

                    # Xác định dựa trên ngưỡng
                    has_face_features = (entropy > 6.5 and contrast > 40 and skin_percent > 0.3)
                    if not has_face_features:
                        print("Failed face verification with entropy/contrast/skin check")
                    return has_face_features

            except Exception as e:
                print(f"Error in face orientation check: {str(e)}")
                return False

        # Đảm bảo tọa độ nằm trong ảnh
        fx = max(0, fx)
        fy = max(0, fy)
        fx2 = min(image.shape[1], fx + fw)
        fy2 = min(image.shape[0], fy + fh)
        fw = fx2 - fx
        fh = fy2 - fy

        # Trích xuất khuôn mặt đã điều chỉnh
        face_img = image[fy:fy + fh, fx:fx + fw]

        # Lưu khuôn mặt để debug
        best_face_path = f"{debug_dir}/best_face.jpg"
        cv2.imwrite(best_face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
        print(f"Best face: {best_face_path}")

        # Kiểm tra nội dung khuôn mặt và hướng
        if not verify_face_orientation(face_img, best_face.get('keypoints', None)):
            print("Face verification failed - detected area is not a frontal face")
            expression_results['debug_info']['reason'] = "Detected area is not a frontal face"
            return expression_results

        # Kiểm tra keypoints nếu có
        if best_face and 'keypoints' in best_face:
            # Lấy keypoints
            keypoints = best_face['keypoints']

            # MTCNN trả về 5 điểm: left_eye, right_eye, nose, mouth_left, mouth_right
            required_points = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
            found_points = [point for point in required_points if point in keypoints]

            # Yêu cầu ít nhất 3 điểm mốc để phân tích biểu cảm
            if len(found_points) < 3:
                print(f"Not enough facial keypoints detected: {len(found_points)}/5")
                expression_results['debug_info']['reason'] = f"Not enough facial keypoints: {len(found_points)}/5"
                return expression_results
            else:
                print(f"Found {len(found_points)}/5 facial keypoints")

        # 5. PHÂN TÍCH BIỂU CẢM SỬ DỤNG DEEPFACE
        try:
            # Kiểm tra và cài đặt DeepFace nếu cần
            try:
                from deepface import DeepFace
            except ImportError:
                print("Installing DeepFace library...")
                os.system("pip install deepface")
                from deepface import DeepFace

            print("Starting emotion analysis with DeepFace...")

            # Lưu ảnh khuôn mặt tạm thời để DeepFace xử lý
            temp_face_path = f"{debug_dir}/temp_face.jpg"
            cv2.imwrite(temp_face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

            # Phân tích cảm xúc với DeepFace với các tùy chọn mở rộng cho ảnh thể thao
            try:
                analysis = DeepFace.analyze(temp_face_path,
                                            actions=['emotion'],
                                            enforce_detection=False,
                                            detector_backend='opencv')
            except:
                print("Retrying with different backend...")
                analysis = DeepFace.analyze(temp_face_path,
                                            actions=['emotion'],
                                            enforce_detection=False,
                                            detector_backend='ssd')

            # Xử lý kết quả
            if isinstance(analysis, list):
                emotion_data = analysis[0]
            else:
                emotion_data = analysis

            dominant_emotion = emotion_data['dominant_emotion']
            emotion_scores = emotion_data['emotion']

            print(f"DeepFace detected emotion: {dominant_emotion}")
            print(f"Emotion scores: {emotion_scores}")

            # Tính cường độ cảm xúc từ điểm số
            dominant_score = emotion_scores[dominant_emotion]
            other_emotions = [v for k, v in emotion_scores.items() if k != dominant_emotion]
            avg_other = sum(other_emotions) / len(other_emotions) if other_emotions else 0
            emotion_intensity = min(0.95, max(0.2, dominant_score / 100))

        except Exception as e:
            print(f"DeepFace analysis failed: {str(e)}")
            print("Falling back to basic emotion analysis...")

            # Fallback to basic analysis if DeepFace fails
            gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray) / 255.0
            contrast = np.std(gray) / 128.0

            # Calculate image histogram (simplified)
            hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            # Calculate image gradient (texture)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mean = np.mean(np.sqrt(sobelx ** 2 + sobely ** 2)) / 128.0

            # Divide image into regions
            h, w = gray.shape
            top = gray[:h // 3, :]  # Forehead/eyes area
            middle = gray[h // 3:2 * h // 3, :]  # Nose area
            bottom = gray[2 * h // 3:, :]  # Mouth/chin area

            # Calculate features by region
            top_contrast = np.std(top) / 128.0
            middle_contrast = np.std(middle) / 128.0
            bottom_contrast = np.std(bottom) / 128.0

            # Initialize emotions
            emotion_scores = {
                'neutral': 0.10,
                'happy': 0.10,
                'surprise': 0.10,
                'sad': 0.10,
                'angry': 0.10,
                'fear': 0.10,
                'disgust': 0.10
            }

            # Analyze brightness & contrast
            if brightness > 0.6:  # Bright -> happy/positive
                emotion_scores['happy'] += 0.15
                emotion_scores['neutral'] -= 0.05
            elif brightness < 0.4:  # Dark -> serious/negative
                emotion_scores['sad'] += 0.10
                emotion_scores['angry'] += 0.05
                emotion_scores['happy'] -= 0.05

            # Analyze texture
            if gradient_mean > 0.25:  # High texture -> strong expression
                emotion_scores['surprise'] += 0.15
                emotion_scores['happy'] += 0.05
                emotion_scores['neutral'] -= 0.10

            # Mouth area analysis
            if bottom_contrast > 0.20:
                if brightness > 0.45:
                    emotion_scores['happy'] += 0.25
                else:
                    emotion_scores['surprise'] += 0.20

            # Ensure no negative values
            emotion_scores = {k: max(0.01, v) for k, v in emotion_scores.items()}

            # Normalize
            total = sum(emotion_scores.values())
            emotion_scores = {k: v / total for k, v in emotion_scores.items()}

            # Get dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            dominant_score = emotion_scores[dominant_emotion]

            # Calculate intensity
            other_scores = [v for k, v in emotion_scores.items() if k != dominant_emotion]
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
        if "action_analysis" in locals() and "action_level" in locals()["action_analysis"]:
            action_level = locals()["action_analysis"]["action_level"]

        # Điều chỉnh biểu cảm dựa trên ngữ cảnh thể thao
        contextual_emotions = emotion_scores.copy()

        # Các môn thể thao đối kháng
        combat_sports = ['Boxing', 'Wrestling', 'Martial Arts']
        team_sports = ['Soccer', 'Basketball', 'Baseball', 'Football', 'Ball Sport']
        track_sports = ['Running', 'Track', 'Sprint', 'Athletics']

        # Phát hiện thể thao điền kinh từ ảnh
        if any(name in str(subject_path).lower() for name in ['track', 'run', 'sprint', 'athlet']):
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

        # Điều chỉnh cho môn điền kinh
        elif sport_type in track_sports or 'Track and Field' in sport_type:
            print(f"Adjusting emotions for track sport")
            # Vận động viên điền kinh thường thể hiện quyết tâm/nỗ lực
            contextual_emotions['angry'] = contextual_emotions.get('angry', 0) * 1.2  # quyết tâm
            contextual_emotions['happy'] = contextual_emotions.get('happy', 0) * 1.1  # phấn khích
            emotion_intensity = min(0.95, emotion_intensity * 1.3)  # cường độ cao

        # Điều chỉnh theo mức độ hành động
        if action_level > 0.7:
            print(f"Adjusting emotions for high action level: {action_level:.2f}")
            # Hành động mạnh mẽ -> biểu cảm mạnh
            emotion_intensity = min(0.95, emotion_intensity * 1.3)

        # Chuẩn hóa lại cảm xúc sau khi điều chỉnh
        total = sum(v for k, v in contextual_emotions.items())
        if total > 0:
            contextual_emotions = {k: v / total for k, v in contextual_emotions.items()}

        # Xác định cảm xúc chính sau điều chỉnh
        adjusted_emotion = max(contextual_emotions, key=contextual_emotions.get)

        print(f"Context-adjusted emotion: {adjusted_emotion} (from {dominant_emotion})")
        print(f"Adjusted intensity: {emotion_intensity:.2f}")

        # Đánh giá giá trị tình cảm
        if emotion_intensity > 0.7:
            emotional_value = 'Very High'
        elif emotion_intensity > 0.5:
            emotional_value = 'High'
        elif emotion_intensity > 0.3:
            emotional_value = 'Medium'
        else:
            emotional_value = 'Low'

        # Xây dựng kết quả cuối cùng
        expression_results = {
            'has_faces': True,
            'dominant_emotion': adjusted_emotion,
            'original_emotion': dominant_emotion,
            'emotion_scores': emotion_scores,
            'contextual_scores': contextual_emotions,
            'emotion_intensity': emotion_intensity,
            'emotional_value': emotional_value,
            'sport_context': sport_type,
            'action_level': action_level,
            'face_coordinates': (fx, fy, fw, fh),
            'face_path': best_face_path,
            'expressions': [{
                'box': (fx, fy, fx + fw, fy + fh),
                'emotion': adjusted_emotion,
                'scores': contextual_emotions
            }]
        }

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

    plt.tight_layout()
    plt.savefig("sports_analysis_results.png", dpi=150)
    plt.show()

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
        f.write(f"- Action level: {action_analysis['action_quality']} ({action_analysis['action_level']:.2f})\n")

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