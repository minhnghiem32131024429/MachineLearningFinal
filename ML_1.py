import os
import traceback

import PIL.Image
from ultralytics.engine import results

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
import math
import matplotlib
matplotlib.use('Agg')  # Sử dụng Agg backend thay vì interactive backend
# Thêm sau các import hiện có
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

# Định nghĩa kết nối giữa các keypoints cho khung xương
POSE_CONNECTIONS = [
    # Đầu
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    # Thân
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Tay
    (5, 7), (7, 9), (6, 8), (8, 10),
    # Chân
    (11, 13), (13, 15), (12, 14), (14, 16)
]

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
        "a soccer ball", "a white and black soccer ball", "a football used in soccer games",
        "a basketball", "an orange basketball with black lines", "a ball used in basketball",
        "a tennis ball", "a yellow-green tennis ball", "a small fuzzy ball used in tennis",
        "a volleyball", "a white volleyball with panels", "a ball used in volleyball games",
        "a baseball", "a white baseball with red stitching", "a small hard ball used in baseball",
        "a golf ball", "a small white golf ball with dimples", "a ball used in golf",
        "a rugby ball", "an oval-shaped rugby ball", "a ball used in rugby"
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

    # Map về tên chuẩn của loại bóng dựa trên mô tả tốt nhất
    if "soccer" in best_match or "football" in best_match:
        return "soccer ball"
    elif "basketball" in best_match:
        return "basketball"
    elif "tennis" in best_match:
        return "tennis ball"
    elif "volleyball" in best_match:
        return "volleyball"
    elif "baseball" in best_match:
        return "baseball"
    elif "golf" in best_match:
        return "golf ball"
    elif "rugby" in best_match:
        return "rugby ball"

    # Nếu không chắc chắn, giữ nguyên nhãn gốc
    return "sports ball"


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
            return detect_faces_fallback(img_bgr)

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
        return detect_faces_fallback(img_bgr)


def detect_faces_fallback(image):
    """
    Phương pháp dự phòng sử dụng Haar Cascade kết hợp với xử lý ảnh
    để cải thiện khả năng phát hiện người da màu và các góc nghiêng
    """
    print("Sử dụng phương pháp dự phòng phát hiện khuôn mặt...")

    # Tạo thư mục cho cascade files
    cascade_dir = "cascade"
    os.makedirs(cascade_dir, exist_ok=True)

    # Đường dẫn tới các file cascade
    cascade_files = {
        "frontal_face": os.path.join(cascade_dir, "haarcascade_frontalface_default.xml"),
        "frontal_face_alt": os.path.join(cascade_dir, "haarcascade_frontalface_alt.xml"),
        "profile_face": os.path.join(cascade_dir, "haarcascade_profileface.xml")
    }

    # Kiểm tra và sao chép cascade files từ OpenCV nếu chưa có
    for name, filepath in cascade_files.items():
        if not os.path.exists(filepath):
            # Tìm trong thư mục cài đặt OpenCV
            cv2_path = os.path.dirname(cv2.__file__)
            cv2_data = os.path.join(cv2_path, 'data')

            # Tên file gốc
            filename = os.path.basename(filepath)
            source_path = os.path.join(cv2_data, filename)

            if os.path.exists(source_path):
                import shutil
                shutil.copy(source_path, filepath)
                print(f"Đã sao chép {filename}")
            else:
                print(f"Không tìm thấy {filename} trong thư viện OpenCV")

    # Tạo các bộ phát hiện
    detectors = {}
    for name, filepath in cascade_files.items():
        if os.path.exists(filepath):
            detectors[name] = cv2.CascadeClassifier(filepath)

    # Nếu không có detector nào, trả về rỗng
    if not detectors:
        print("Không tìm thấy bộ phát hiện khuôn mặt!")
        return []

    # Chuyển ảnh sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Cải thiện độ tương phản để xử lý tốt hơn với da màu
    gray = cv2.equalizeHist(gray)

    # Phát hiện khuôn mặt từ nhiều góc
    faces = []

    # 1. Phát hiện khuôn mặt chính diện
    if "frontal_face" in detectors:
        frontal_faces = detectors["frontal_face"].detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(frontal_faces) > 0:
            for (x, y, w, h) in frontal_faces:
                faces.append((x, y, w, h))

    # 2. Thử với frontal_face_alt nếu không phát hiện được
    if len(faces) == 0 and "frontal_face_alt" in detectors:
        alt_faces = detectors["frontal_face_alt"].detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )

        if len(alt_faces) > 0:
            for (x, y, w, h) in alt_faces:
                faces.append((x, y, w, h))

    # 3. Thử với profile_face nếu vẫn không phát hiện được
    if len(faces) == 0 and "profile_face" in detectors:
        # Phát hiện khuôn mặt nghiêng
        profile_faces = detectors["profile_face"].detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )

        if len(profile_faces) > 0:
            for (x, y, w, h) in profile_faces:
                faces.append((x, y, w, h))

        # Thử phát hiện khuôn mặt nghiêng về phía bên kia (lật ảnh)
        flipped = cv2.flip(gray, 1)
        flipped_profile_faces = detectors["profile_face"].detectMultiScale(
            flipped,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )

        if len(flipped_profile_faces) > 0:
            img_width = gray.shape[1]
            for (x, y, w, h) in flipped_profile_faces:
                # Điều chỉnh tọa độ x cho ảnh đã lật
                faces.append((img_width - x - w, y, w, h))

    return faces


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

    # Phân tích hành động của vận động viên chính từ pose data
    if 'pose_analysis' in sports_analysis and isinstance(sports_analysis['pose_analysis'], dict) and 'poses' in \
            sports_analysis['pose_analysis']:
        poses = sports_analysis['pose_analysis']['poses']
        if poses and len(poses) > 0:
            # Phân tích hành động dựa trên pose đầu tiên (hoặc pose được lọc theo main subject)
            main_pose = poses[0]
            action_analysis = analyze_athlete_action(main_pose)
            sports_analysis['action_analysis'] = action_analysis
            print(f"Phân tích hành động: {action_analysis['action']} - {action_analysis['description']}")

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


def detect_human_pose(img_data, conf_threshold=0.15, main_subject_box=None):
    print("DEBUG - Bắt đầu phát hiện pose")
    """
    Sử dụng YOLOv8-Pose để phát hiện các keypoint trên cơ thể người

    Args:
        img_data: Dict chứa ảnh gốc và ảnh đã resize
        conf_threshold: Ngưỡng confidence cho việc phát hiện

    Returns:
        Dict: Thông tin về pose các người được phát hiện
    """
    if not POSE_MODEL_AVAILABLE:
        return {"poses": []}

    # Tạo cấu trúc kết quả
    pose_results = {
        "poses": [],
        "keypoint_names": {
            0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
            5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
            9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
            13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
        }
    }

    # Load model (chỉ tải một lần)
    if not hasattr(detect_human_pose, 'model'):
        print("Đang tải model YOLOv8-Pose...")
        detect_human_pose.model = YOLO('yolov8x-pose-p6.pt')  # model nhỏ
        print("Đã tải model YOLOv8-Pose thành công")

    # Dự đoán trên ảnh
    results = detect_human_pose.model(img_data['resized_array'])

    # Chỉ lấy kết quả đầu tiên
    for result in results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.data
            for person_id, person_keypoints in enumerate(keypoints):
                # Tạo dict chứa thông tin pose của mỗi người
                person_pose = {
                    "person_id": person_id,
                    "keypoints": [],
                    "bbox": None
                }

                # Lấy keypoint và confidence
                for kp_id, kp in enumerate(person_keypoints):
                    x, y, conf = kp.tolist()
                    if conf >= conf_threshold:
                        person_pose["keypoints"].append({
                            "id": kp_id,
                            "name": pose_results["keypoint_names"].get(kp_id, f"kp_{kp_id}"),
                            "x": float(x),
                            "y": float(y),
                            "confidence": float(conf)
                        })

                # Tính bounding box từ keypoints
                if person_pose["keypoints"]:
                    kp_coords = [(kp["x"], kp["y"]) for kp in person_pose["keypoints"]]
                    x_min = min([x for x, _ in kp_coords])
                    y_min = min([y for _, y in kp_coords])
                    x_max = max([x for x, _ in kp_coords])
                    y_max = max([y for _, y in kp_coords])
                    person_pose["bbox"] = [x_min, y_min, x_max, y_max]

                # Thêm vào kết quả
                if person_pose["keypoints"]:
                    pose_results["poses"].append(person_pose)


    print(f"DEBUG - pose_results có số poses: {len(pose_results.get('poses', []))}")
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
    composition = analysis["composition_analysis"] if "composition_analysis" in analysis else {}

    # Phân tích môi trường thể thao
    depth_map = None
    if 'depth_map' in analysis:
        depth_map = analysis['depth_map']

    env_analysis = analyze_sports_environment(img_data, depth_map)

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
        'soccer ball': 'Soccer',  # Thêm
        'basketball': 'Basketball',  # Thêm
        'volleyball': 'Volleyball',  # Thêm
        'baseball': 'Baseball',  # Thêm
        'baseball bat': 'Baseball',
        'baseball glove': 'Baseball',
        'golf ball': 'Golf',  # Thêm
        'rugby ball': 'Rugby',  # Thêm
        'skateboard': 'Skateboarding',
        'surfboard': 'Surfing',
        'frisbee': 'Frisbee',
        'skis': 'Skiing',
        'snowboard': 'Snowboarding'
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


def analyze_athlete_action(pose_data):
    """
    Phân tích hành động của vận động viên dựa trên dữ liệu pose.

    Args:
        pose_data (dict): Dữ liệu pose của vận động viên chính

    Returns:
        dict: Kết quả phân tích hành động và độ tin cậy
    """
    if not pose_data or 'keypoints' not in pose_data:
        return {'action': 'unknown', 'confidence': 0, 'description': 'Không đủ dữ liệu pose'}

    # Lấy các keypoints
    keypoints = {}
    for kp in pose_data['keypoints']:
        keypoints[kp['id']] = {'x': kp['x'], 'y': kp['y'], 'conf': kp['confidence']}

    # Tính các vector chuyển động và góc cho phân tích
    vectors = {}
    angles = {}

    # Kiểm tra đủ keypoints để phân tích
    required_keypoints = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # vai, khuỷu, cổ tay, hông, đầu gối, cổ chân
    if not all(i in keypoints for i in required_keypoints):
        return {'action': 'standing', 'confidence': 0.3, 'description': 'Đang đứng hoặc tư thế tĩnh'}

    # Tính góc khuỷu tay
    try:
        # Góc khuỷu tay phải (vai - khuỷu - cổ tay)
        r_elbow_angle = calculate_angle(
            [keypoints[6]['x'], keypoints[6]['y']],  # vai phải
            [keypoints[8]['x'], keypoints[8]['y']],  # khuỷu phải
            [keypoints[10]['x'], keypoints[10]['y']]  # cổ tay phải
        )

        # Góc khuỷu tay trái
        l_elbow_angle = calculate_angle(
            [keypoints[5]['x'], keypoints[5]['y']],  # vai trái
            [keypoints[7]['x'], keypoints[7]['y']],  # khuỷu trái
            [keypoints[9]['x'], keypoints[9]['y']]  # cổ tay trái
        )

        # Góc đầu gối phải
        r_knee_angle = calculate_angle(
            [keypoints[12]['x'], keypoints[12]['y']],  # hông phải
            [keypoints[14]['x'], keypoints[14]['y']],  # đầu gối phải
            [keypoints[16]['x'], keypoints[16]['y']]  # cổ chân phải
        )

        # Góc đầu gối trái
        l_knee_angle = calculate_angle(
            [keypoints[11]['x'], keypoints[11]['y']],  # hông trái
            [keypoints[13]['x'], keypoints[13]['y']],  # đầu gối trái
            [keypoints[15]['x'], keypoints[15]['y']]  # cổ chân trái
        )

        # Tính chiều cao của người (khoảng cách từ mũi đến mắt cá)
        height_ratio = 0

        if 0 in keypoints and 15 in keypoints and 16 in keypoints:
            nose_y = keypoints[0]['y']
            avg_ankle_y = (keypoints[15]['y'] + keypoints[16]['y']) / 2
            person_height = abs(avg_ankle_y - nose_y)

            # Tính độ cao của trọng tâm so với chiều cao
            hip_y = (keypoints[11]['y'] + keypoints[12]['y']) / 2
            height_ratio = (avg_ankle_y - hip_y) / person_height

        # Lưu các góc tính được
        angles = {
            'r_elbow': r_elbow_angle,
            'l_elbow': l_elbow_angle,
            'r_knee': r_knee_angle,
            'l_knee': l_knee_angle,
            'height_ratio': height_ratio
        }

        # Xác định hành động dựa vào góc của các khớp
        return determine_action(angles)

    except Exception as e:
        print(f"Lỗi khi phân tích hành động: {e}")
        return {'action': 'unknown', 'confidence': 0, 'description': f'Lỗi phân tích: {str(e)}'}


def calculate_angle(a, b, c):
    """Tính góc giữa ba điểm (tính bằng độ)"""
    # Vector AB và BC
    ab = [b[0] - a[0], b[1] - a[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    # Tích vô hướng
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]

    # Độ dài vector
    ab_magnitude = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    bc_magnitude = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    # Tính góc
    if ab_magnitude * bc_magnitude == 0:
        return 0

    cos_angle = dot_product / (ab_magnitude * bc_magnitude)
    cos_angle = max(-1, min(1, cos_angle))  # Đảm bảo giá trị nằm trong [-1, 1]
    angle = math.degrees(math.acos(cos_angle))

    return angle


def determine_action(angles):
    """Xác định hành động dựa vào các góc khớp"""
    r_elbow = angles['r_elbow']
    l_elbow = angles['l_elbow']
    r_knee = angles['r_knee']
    l_knee = angles['l_knee']
    height_ratio = angles['height_ratio']

    # Phát hiện chạy: Một chân duỗi thẳng, một chân gập
    if ((r_knee < 120 and l_knee > 150) or (l_knee < 120 and r_knee > 150)) and height_ratio > 0.4:
        return {
            'action': 'running',
            'confidence': 0.85,
            'description': 'Đang chạy',
            'angles': angles
        }

    # Phát hiện nhảy: Cả hai chân đều gập, thân người cao hơn bình thường
    if r_knee < 110 and l_knee < 110 and height_ratio > 0.6:
        return {
            'action': 'jumping',
            'confidence': 0.8,
            'description': 'Đang nhảy',
            'angles': angles
        }

    # Phát hiện nhồi bóng (dribbling): Một tay gập mạnh, thân người nghiêng
    if (r_elbow < 100 or l_elbow < 100) and height_ratio > 0.45:
        return {
            'action': 'dribbling',
            'confidence': 0.7,
            'description': 'Đang nhồi bóng',
            'angles': angles
        }

    # Phát hiện ném/sút: Tay duỗi thẳng, góc lớn
    if (r_elbow > 160 or l_elbow > 160) and (r_knee > 150 or l_knee > 150):
        return {
            'action': 'throwing',
            'confidence': 0.75,
            'description': 'Đang ném/sút',
            'angles': angles
        }

    # Phát hiện tư thế phòng thủ: Chân rộng, gập nhẹ
    if 110 < r_knee < 150 and 110 < l_knee < 150 and 0.3 < height_ratio < 0.5:
        return {
            'action': 'defending',
            'confidence': 0.65,
            'description': 'Tư thế phòng thủ',
            'angles': angles
        }

    # Mở rộng với các hành động khác...
    # Có thể thêm các hành động như bóng đá, bóng chuyền, tennis, etc.

    # Mặc định trả về tư thế đứng
    return {
        'action': 'standing',
        'confidence': 0.5,
        'description': 'Đang đứng hoặc tư thế chưa xác định',
        'angles': angles
    }


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


def create_caption_visualization(caption, width):
    pass


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
            main_center_x = (main_x1 + main_x2) / 2
            main_center_y = (main_y1 + main_y2) / 2

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

    # THÊM MỚI: Xử lý và hiển thị caption
    if caption:
        # Đọc ảnh kết quả đã lưu
        result_img = cv2.imread("sports_analysis_results.png")
        if result_img is None:
            print("Lỗi: Không thể đọc ảnh kết quả phân tích")
            return

        # Tạo hình ảnh caption trực quan đẹp mắt
        caption_img = create_caption_visualization(caption, width=result_img.shape[1])

        # Ghép dọc với kết quả phân tích
        combined = np.vstack([result_img, caption_img])
        cv2.imwrite("sports_analysis_with_caption.png", combined)

        # Lưu thêm caption riêng cho tiện sử dụng
        cv2.imwrite("sports_caption.png", caption_img)

        print(f"\nCaption: {caption}")
        print("Visualization with caption saved as: sports_analysis_with_caption.png")

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

        # Lưu caption
        if caption:
            f.write("\nImage Caption:\n")
            f.write(caption + "\n")


def analyze_sports_image(file_path):
    """Main function to analyze sports images"""
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

    # Step 6.5: THÊM PHÁT HIỆN POSE
    print("Detecting human poses...")
    pose_results = detect_human_pose(img_data, conf_threshold=0.15)
    # Cập nhật sports_analysis với pose_results
    sports_analysis['pose_analysis'] = pose_results
    print(f"DEBUG B - sports_analysis sau khi gán pose: {sports_analysis.keys()}")

    # Tạo kết quả phân tích cuối cùng
    analysis_result = {
        'detections': detections,
        'sports_analysis': sports_analysis,  # sports_analysis đã có pose_analysis
        'action_analysis': action_analysis,
        'composition_analysis': composition_analysis,
        'facial_analysis': facial_analysis
    }

    # Step 7: Visualize results với hiển thị biểu cảm cải tiến
    print("Visualizing results...")
    print(f"DEBUG C - ID của sports_analysis trước khi visualize: {id(sports_analysis)}")
    visualize_sports_results(img_data, detections, depth_map,
                             sports_analysis, action_analysis, composition_analysis,
                             facial_analysis)

    t_end = time.time()
    print(f"\nAnalysis completed in {t_end - t_start:.2f} seconds")

    # Tạo caption từ kết quả phân tích
    caption = generate_sports_caption(analysis_result)
    print(f"\nCaption: {caption}")

    # Thêm caption vào kết quả trả về
    analysis_result['caption'] = caption

    return analysis_result


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
    sport_type = composition_analysis.get('sport_type', 'Unknown').lower()
    sport_type_original = composition_analysis.get('sport_type', 'Unknown')
    athlete_count = detections.get('athletes', 0)
    action_level = action_analysis.get('action_level', 0)
    action_quality = action_analysis.get('action_quality', '')
    equipment = action_analysis.get('equipment_types', [])

    # Sport-specific terminology
    sport_specific_terms = {
        'soccer': ['match', 'pitch', 'soccer', 'football', 'kick', 'goal'],
        'football': ['stadium', 'team', 'touchdown', 'offense', 'quarterback'],
        'basketball': ['court', 'shot', 'hoop', 'dunk', 'basket'],
        'tennis': ['court', 'player', 'serve', 'set', 'stroke', 'match point'],
        'baseball': ['field', 'hit', 'home run', 'pitcher', 'batter'],
        'swimming': ['pool', 'lane', 'swimmer', 'stroke', 'race'],
        'volleyball': ['net', 'court', 'spike', 'serve'],
        'track': ['track', 'race', 'sprinter', 'athlete'],
        'running': ['race', 'track', 'runner', 'sprint'],
        'boxing': ['ring', 'boxer', 'punch', 'match', 'bout'],
        'skiing': ['snow', 'slope', 'skier', 'mountain'],
        'skating': ['ice', 'skater', 'performance', 'rink'],
        'surfing': ['wave', 'beach', 'surfer', 'ocean'],
        'skateboarding': ['skate park', 'skateboarder', 'trick', 'jump'],
        'golf': ['course', 'club', 'swing', 'golfer', 'hole'],
        'rugby': ['field', 'tackle', 'player', 'scrum'],
        'martial arts': ['mat', 'fighter', 'technique', 'match']
    }

    # Determine sport type based on name and equipment
    detected_sport = 'unknown'
    sport_confidence = 0

    for sport_name, terms in sport_specific_terms.items():
        # Check sport name
        if sport_name in sport_type.lower():
            detected_sport = sport_name
            sport_confidence = 0.8
            break

        # Check equipment
        for eq in equipment:
            eq_lower = eq.lower()
            if sport_name in eq_lower or any(term.lower() in eq_lower for term in terms):
                detected_sport = sport_name
                sport_confidence = 0.6
                break

    # If no specific sport found but "sports ball" is detected
    if detected_sport == 'unknown' and any('ball' in eq.lower() for eq in equipment):
        detected_sport = 'ball sport'
        sport_confidence = 0.4

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
        if action_quality == 'High':
            if detected_sport in ['soccer', 'football', 'basketball', 'volleyball']:
                action_options = [
                    "during an intensely competitive play",
                    "at a crucial moment in the match",
                    "in a high-stakes game situation",
                    "executing an advanced technical maneuver",
                    "during a powerful offensive drive"
                ]
            elif detected_sport in ['tennis', 'baseball', 'golf']:
                action_options = [
                    "during a perfectly executed swing",
                    "demonstrating exceptional technique",
                    "at the critical moment of impact",
                    "showcasing masterful control",
                    "with professional athletic form"
                ]
            elif detected_sport in ['running', 'track', 'swimming']:
                action_options = [
                    "at the moment of breakthrough acceleration",
                    "displaying extraordinary effort",
                    "at a decisive point in the race",
                    "during a powerful acceleration phase",
                    "with remarkable concentration and form"
                ]
            else:
                action_options = [
                    "at the peak moment of performance",
                    "with impressive competitive intensity",
                    "during a critical action sequence",
                    "demonstrating professional-level skill",
                    "in a dynamic display of athleticism"
                ]
        elif action_quality == 'Medium':
            action_options = [
                "during active competition",
                "with focused engagement in the event",
                "amidst the flow of the game",
                "demonstrating solid technique",
                "in a noteworthy moment of play"
            ]
        else:
            action_options = [
                "during a moment of calculated preparation",
                "in a strategic positioning phase",
                "during a brief respite in the action",
                "with focused pre-action concentration",
                "before initiating the next movement"
            ]

        action_phrases.append(random.choice(action_options))

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