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

    # MỞ RỘNG DANH SÁCH CÁC LOẠI BÓNG CLIP CÓ THỂ PHÂN LOẠI
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
    # Kiểm tra xem có action boxing nào không
    if 'detected_actions' in img_data:
        for action in img_data['detected_actions']:
            boxing_actions = ['straight_punch', 'left_hook', 'right_hook', 'uppercut', 'body_shot', 'defensive_guard']
            if action['action'] in boxing_actions and action.get('confidence', 0) > 0.5:
                print(f"💥 FORCE BOXING từ analyze_sports_environment: {action['action']}")
                return {'sport_type': 'Boxing', 'confidence': 0.99,
                        'sport_type_source': f'boxing_action_{action["action"]}',
                        'environment_indicators': {'boxing_ring': 0.8}}
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
                for kp_id, kp in enumerate(person_keypoints):
                    x, y, conf = kp.tolist()
                    # Sử dụng threshold linh hoạt cho các keypoints quan trọng
                    dynamic_threshold = conf_threshold
                    if kp_id in [5, 6, 11, 12, 13, 14, 15, 16]:  # Keypoints quan trọng cho thể thao
                        dynamic_threshold = max(0.08, conf_threshold - 0.05)

                    if conf >= dynamic_threshold:
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

def detect_sports_actions(pose_data, sport_type, image_shape, detected_equipment=None, environment_sport=None):
    """
    Phát hiện hành động thể thao cụ thể dựa trên pose keypoints với equipment filtering

    Args:
        pose_data: Dữ liệu pose từ detect_human_pose
        sport_type: Loại thể thao được phát hiện
        image_shape: Kích thước ảnh (height, width)
        detected_equipment: List các equipment đã phát hiện
        environment_sport: Sport type từ environment analysis

    Returns:
        Dict: Thông tin về hành động được phát hiện
    """
    if not pose_data or 'poses' not in pose_data or not pose_data['poses']:
        return {'detected_actions': [], 'confidence': 0.0, 'details': 'No pose data available'}
    # EQUIPMENT FILTERING LOGIC
    equipment_to_sport = {
        'tennis racket': 'tennis',
        'tennis ball': 'tennis',
        'soccer ball': 'soccer',
        'basketball': 'basketball',
        'volleyball': 'volleyball',
        'baseball bat': 'baseball',
        'baseball glove': 'baseball',
        'baseball': 'baseball',
        'american football': 'football',
        'golf ball': 'golf',
        'golf club': 'golf',
        'boxing gloves': 'boxing',
        'ping pong ball': 'table tennis',
        'badminton racket': 'badminton',
        'skis': 'skiing',
        'snowboard': 'snowboarding',
        'surfboard': 'surfing',
        'skateboard': 'skateboarding',
        'frisbee': 'frisbee'
    }

    # Xác định sport từ equipment
    equipment_detected_sport = None
    if detected_equipment:
        for equipment in detected_equipment:
            if equipment.lower() in equipment_to_sport:
                equipment_detected_sport = equipment_to_sport[equipment.lower()]
                print(f"Equipment-based sport detection: {equipment} -> {equipment_detected_sport}")
                break

    # Kiểm tra swimming từ environment (đặc biệt)
    is_swimming_environment = False
    if environment_sport and 'swimming' in environment_sport.lower():
        is_swimming_environment = True
        print("Swimming detected from environment analysis")

    # FILTERING RULES - SỬA LẠI:
    # 1. Nếu có equipment, chỉ detect action của sport đó
    # 2. Nếu environment detect swimming, thêm swimming vào allowed
    # 3. Nếu không có equipment, cho phép sports không cần equipment + swimming nếu có

    allowed_sports = []

    if equipment_detected_sport:
        # Chỉ cho phép sport từ equipment
        allowed_sports = [equipment_detected_sport]
        # NHƯNG vẫn cho phép swimming nếu environment detect
        if is_swimming_environment:
            allowed_sports.append('swimming')
        print(f"Action detection limited to: {allowed_sports}")
    else:
        # Không có equipment - cho phép sports không cần equipment cụ thể
        allowed_sports = ['soccer', 'basketball', 'volleyball', 'running', 'track', 'boxing', 'martial arts']
        # LUÔN LUÔN thêm swimming nếu environment detect
        if is_swimming_environment:
            allowed_sports.append('swimming')
        print(f"No equipment detected - allowed sports: {allowed_sports}")

    height, width = image_shape[:2]
    detected_actions = []

    # Lấy pose chính (thường là pose đầu tiên hoặc có nhiều keypoints nhất)
    main_pose = None
    max_keypoints = 0

    for pose in pose_data['poses']:
        if len(pose.get('keypoints', [])) > max_keypoints:
            max_keypoints = len(pose.get('keypoints', []))
            main_pose = pose

    if not main_pose or not main_pose.get('keypoints'):
        return {'detected_actions': [], 'confidence': 0.0, 'details': 'No valid pose found'}

    # Tạo dict keypoints để dễ truy cập
    kp_dict = {}
    for kp in main_pose['keypoints']:
        if kp['confidence'] > 0.3:  # Chỉ lấy keypoints có độ tin cậy cao
            kp_dict[kp['id']] = {
                'x': kp['x'],
                'y': kp['y'],
                'confidence': kp['confidence'],
                'name': kp['name']
            }

    # Định nghĩa keypoints theo COCO format
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

    def calculate_angle(p1, p2, p3):
        """Tính góc tại điểm p2 giữa p1-p2-p3"""
        try:
            import math
            v1 = (p1['x'] - p2['x'], p1['y'] - p2['y'])
            v2 = (p3['x'] - p2['x'], p3['y'] - p2['y'])

            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

            if mag1 == 0 or mag2 == 0:
                return 0

            cos_angle = dot_product / (mag1 * mag2)
            cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
            angle = math.acos(cos_angle) * 180 / math.pi
            return angle
        except:
            return 0

    def get_body_orientation():
        """Xác định hướng của cơ thể (trước/sau/trái/phải)"""
        orientation = "unknown"

        # Kiểm tra vai
        if 5 in kp_dict and 6 in kp_dict:  # left_shoulder, right_shoulder
            left_shoulder = kp_dict[5]
            right_shoulder = kp_dict[6]
            shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])

            # Nếu vai rộng -> nhìn trước/sau
            # Nếu vai hẹp -> nhìn nghiêng
            if shoulder_width > width * 0.15:  # Vai rộng
                # Kiểm tra mũi để xác định trước/sau
                if 0 in kp_dict:  # nose
                    nose = kp_dict[0]
                    shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
                    nose_offset = abs(nose['x'] - shoulder_center_x)

                    if nose_offset < width * 0.05:  # Mũi ở giữa
                        orientation = "front"
                    else:
                        orientation = "front_angled"
                else:
                    orientation = "front"
            else:  # Vai hẹp - nhìn nghiêng
                # Xác định trái/phải dựa trên vị trí vai
                if left_shoulder['x'] < right_shoulder['x']:
                    orientation = "left_side"
                else:
                    orientation = "right_side"

        return orientation

    # Xác định hướng cơ thể
    body_orientation = get_body_orientation()

    # PHÂN TÍCH HÀNH ĐỘNG THEO TỪNG MÔN THỂ THAO
    sport_lower = sport_type.lower()

    # ==================== BÓNG ĐÁ (SOCCER) ====================
    if ('soccer' in sport_lower or 'football' in sport_lower) and 'soccer' in allowed_sports:
        action_confidence = 0.0

        # 1. PRE-KICK STANCE - Tư thế chuẩn bị đá bóng
        def detect_pre_kick_stance():
            confidence = 0.0
            details = []

            # Kiểm tra chân trụ và chân đá
            if (11 in kp_dict and 12 in kp_dict and 13 in kp_dict and 14 in kp_dict and
                    15 in kp_dict and 16 in kp_dict):

                left_hip = kp_dict[11]
                right_hip = kp_dict[12]
                left_knee = kp_dict[13]
                right_knee = kp_dict[14]
                left_ankle = kp_dict[15]
                right_ankle = kp_dict[16]

                # Tính khoảng cách giữa hai chân (stance width)
                ankle_distance = abs(left_ankle['x'] - right_ankle['x'])
                hip_width = abs(left_hip['x'] - right_hip['x'])

                # Stance rộng hơn bình thường (chuẩn bị đá)
                wide_stance = ankle_distance > hip_width * 1.5

                # Kiểm tra trọng tâm nghiêng (weight shift)
                # Chân trụ thẳng, chân đá hơi gấp
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                # Một chân thẳng (supporting leg), một chân hơi gấp (kicking leg)
                straight_leg = left_knee_angle > 160 or right_knee_angle > 160
                bent_leg = left_knee_angle < 140 or right_knee_angle < 140

                # Chân đá hơi nâng hoặc lùi về phía sau
                if left_knee_angle < right_knee_angle:  # Left leg is kicking leg
                    kicking_leg_back = left_ankle['x'] < right_ankle['x'] - width * 0.05
                    kicking_leg_lifted = left_ankle['y'] < right_ankle['y']
                    kicking_side = "left"
                else:  # Right leg is kicking leg
                    kicking_leg_back = right_ankle['x'] > left_ankle['x'] + width * 0.05
                    kicking_leg_lifted = right_ankle['y'] < left_ankle['y']
                    kicking_side = "right"

                # Tính điểm confidence
                score = 0.0
                if wide_stance:
                    score += 0.3
                    details.append("wide stance detected")

                if straight_leg and bent_leg:
                    score += 0.4
                    details.append("supporting leg positioned")

                if kicking_leg_back or kicking_leg_lifted:
                    score += 0.3
                    details.append(f"{kicking_side} leg in pre-kick position")

                confidence = min(0.85, score)

            return confidence, details

        # 2. SHOOTING/KICKING - Đang thực hiện cú đá
        def detect_shooting():
            confidence = 0.0
            details = []

            if 13 in kp_dict and 15 in kp_dict and 11 in kp_dict:  # left_knee, left_ankle, left_hip
                left_knee = kp_dict[13]
                left_ankle = kp_dict[15]
                left_hip = kp_dict[11]

                # Tính góc chân trái
                knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

                # Kiểm tra chân có đang duỗi không (shooting motion)
                if knee_angle > 140:  # Chân duỗi
                    # Kiểm tra độ cao của chân
                    if left_ankle['y'] < left_hip['y']:  # Chân nâng cao
                        confidence = 0.8
                        details.append(f'Left leg extended for shot (angle: {knee_angle:.1f}°)')

            # Kiểm tra chân phải tương tự
            if 14 in kp_dict and 16 in kp_dict and 12 in kp_dict:  # right_knee, right_ankle, right_hip
                right_knee = kp_dict[14]
                right_ankle = kp_dict[16]
                right_hip = kp_dict[12]

                knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                if knee_angle > 140 and right_ankle['y'] < right_hip['y']:
                    confidence = max(confidence, 0.8)
                    details.append(f'Right leg extended for shot (angle: {knee_angle:.1f}°)')

            return confidence, details

        # 3. APPROACH RUN - Chạy tới để đá bóng
        def detect_approach_run():
            confidence = 0.0
            details = []

            if (13 in kp_dict and 14 in kp_dict and 15 in kp_dict and 16 in kp_dict and
                    5 in kp_dict and 11 in kp_dict):

                left_knee = kp_dict[13]
                right_knee = kp_dict[14]
                shoulder = kp_dict[5]
                hip = kp_dict[11]

                # Kiểm tra bước chạy (một chân nâng cao)
                ground_level = height * 0.9
                high_knee = (left_knee['y'] < ground_level * 0.8 or
                             right_knee['y'] < ground_level * 0.8)

                # Cơ thể nghiêng về phía trước (running lean)
                forward_lean = shoulder['y'] < hip['y'] - height * 0.02

                if high_knee and forward_lean:
                    confidence = 0.7
                    details.append("Running approach with forward lean")

            return confidence, details

        # 4. DRIBBLING - Giữ bóng, di chuyển
        def detect_dribbling():
            confidence = 0.0
            details = []

            if (13 in kp_dict and 15 in kp_dict and 14 in kp_dict and 16 in kp_dict and
                    5 in kp_dict and 11 in kp_dict):

                left_ankle = kp_dict[15]
                right_ankle = kp_dict[16]
                shoulder = kp_dict[5]
                hip = kp_dict[11]

                # Cả hai chân gần đất (controlling stance)
                ground_level = height * 0.9
                both_feet_low = (left_ankle['y'] > ground_level * 0.85 and
                                 right_ankle['y'] > ground_level * 0.85)

                # Cơ thể hơi nghiêng để kiểm soát bóng
                slight_lean = abs(shoulder['y'] - hip['y']) < height * 0.08

                if both_feet_low and slight_lean:
                    confidence = 0.7
                    details.append('Low stance with controlled ball movement')

            return confidence, details

        # Chạy tất cả detections
        pre_kick_conf, pre_kick_details = detect_pre_kick_stance()
        shooting_conf, shooting_details = detect_shooting()
        approach_conf, approach_details = detect_approach_run()
        dribbling_conf, dribbling_details = detect_dribbling()

        # Ưu tiên theo thứ tự: shooting > pre_kick > approach > dribbling
        if shooting_conf > 0.6:
            action_confidence = shooting_conf
            detected_actions.append({
                'action': 'shooting',
                'confidence': shooting_conf,
                'details': '; '.join(shooting_details),
                'body_part': 'legs'
            })
        elif pre_kick_conf > 0.5:
            action_confidence = pre_kick_conf
            detected_actions.append({
                'action': 'pre_kick_stance',
                'confidence': pre_kick_conf,
                'details': '; '.join(pre_kick_details),
                'body_part': 'full_body'
            })
        elif approach_conf > 0.5:
            action_confidence = approach_conf
            detected_actions.append({
                'action': 'approach_run',
                'confidence': approach_conf,
                'details': '; '.join(approach_details),
                'body_part': 'full_body'
            })
        elif dribbling_conf > 0.5:
            action_confidence = dribbling_conf
            detected_actions.append({
                'action': 'dribbling',
                'confidence': dribbling_conf,
                'details': '; '.join(dribbling_details),
                'body_part': 'full_body'
            })

    # ==================== BÓNG RỔ (BASKETBALL) ====================
    elif 'basketball' in sport_lower and 'basketball' in allowed_sports:
        action_confidence = 0.0

        # 1. SHOOTING - Tay nâng cao, khuỷu tay gấp
        if (9 in kp_dict and 10 in kp_dict and 7 in kp_dict and 8 in kp_dict and
                5 in kp_dict and 6 in kp_dict):

            left_wrist = kp_dict[9]
            right_wrist = kp_dict[10]
            left_elbow = kp_dict[7]
            right_elbow = kp_dict[8]
            left_shoulder = kp_dict[5]
            right_shoulder = kp_dict[6]

            # Kiểm tra tay có nâng cao không
            avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            hands_raised = (left_wrist['y'] < avg_shoulder_y or right_wrist['y'] < avg_shoulder_y)

            if hands_raised:
                # Tính góc khuỷu tay
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                # Góc khuỷu trong khoảng shooting (60-120 độ)
                shooting_angles = (60 <= left_elbow_angle <= 120 or 60 <= right_elbow_angle <= 120)

                if shooting_angles:
                    action_confidence = 0.85
                    detected_actions.append({
                        'action': 'shooting',
                        'confidence': action_confidence,
                        'details': f'Arms raised for shot (L: {left_elbow_angle:.1f}°, R: {right_elbow_angle:.1f}°)',
                        'body_part': 'arms'
                    })

        # 2. DRIBBLING - Một tay xuống thấp, cơ thể cúi
        if (9 in kp_dict and 10 in kp_dict and 5 in kp_dict and 11 in kp_dict):
            left_wrist = kp_dict[9]
            right_wrist = kp_dict[10]
            shoulder = kp_dict[5]
            hip = kp_dict[11]

            # Một tay thấp hơn hông
            low_hand = (left_wrist['y'] > hip['y'] or right_wrist['y'] > hip['y'])

            # Cơ thể hơi cúi
            body_lean = shoulder['y'] < hip['y'] + height * 0.1

            if low_hand and body_lean:
                action_confidence = max(action_confidence, 0.75)
                detected_actions.append({
                    'action': 'dribbling',
                    'confidence': 0.75,
                    'details': 'Low hand position with controlled dribble stance',
                    'body_part': 'arms'
                })

    # ==================== TENNIS ====================
    elif 'tennis' in sport_lower and 'tennis' in allowed_sports:
        action_confidence = 0.0

        # 1. SERVING - Một tay nâng cao (lempar bóng), tay kia chuẩn bị đánh
        if (9 in kp_dict and 10 in kp_dict and 7 in kp_dict and 8 in kp_dict and
                5 in kp_dict and 6 in kp_dict):

            left_wrist = kp_dict[9]
            right_wrist = kp_dict[10]
            left_shoulder = kp_dict[5]
            right_shoulder = kp_dict[6]

            avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2

            # Kiểm tra một tay nâng rất cao (lempar bóng)
            left_very_high = left_wrist['y'] < avg_shoulder_y - height * 0.15
            right_very_high = right_wrist['y'] < avg_shoulder_y - height * 0.15

            if left_very_high or right_very_high:
                action_confidence = 0.8
                serving_hand = "left" if left_very_high else "right"
                detected_actions.append({
                    'action': 'serving',
                    'confidence': action_confidence,
                    'details': f'{serving_hand.title()} hand raised for ball toss',
                    'body_part': f'{serving_hand}_arm'
                })

        # 2. FOREHAND/BACKHAND - Tay duỗi ra một bên
        if (9 in kp_dict and 10 in kp_dict and 5 in kp_dict and 6 in kp_dict):
            left_wrist = kp_dict[9]
            right_wrist = kp_dict[10]
            left_shoulder = kp_dict[5]
            right_shoulder = kp_dict[6]

            shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2

            # Kiểm tra tay duỗi ra xa khỏi cơ thể
            left_extended = abs(left_wrist['x'] - shoulder_center_x) > width * 0.15
            right_extended = abs(right_wrist['x'] - shoulder_center_x) > width * 0.15

            if left_extended or right_extended:
                action_confidence = max(action_confidence, 0.7)
                stroke_type = "forehand" if (
                            left_extended and body_orientation in ["front", "left_side"]) else "backhand"
                detected_actions.append({
                    'action': stroke_type,
                    'confidence': 0.7,
                    'details': f'Arm extended for {stroke_type} stroke',
                    'body_part': 'arms'
                })

    # ==================== BÓNG CHUYỀN (VOLLEYBALL) ====================
    elif 'volleyball' in sport_lower and 'volleyball' in allowed_sports:
        action_confidence = 0.0

        # 1. SPIKING - Nhiều biến thể tư thế tay
        def detect_spiking_variations():
            confidence = 0.0
            details = []
            spike_type = "unknown"

            if (9 in kp_dict and 10 in kp_dict and 7 in kp_dict and 8 in kp_dict and
                    5 in kp_dict and 6 in kp_dict):

                left_wrist = kp_dict[9]
                right_wrist = kp_dict[10]
                left_elbow = kp_dict[7]
                right_elbow = kp_dict[8]
                left_shoulder = kp_dict[5]
                right_shoulder = kp_dict[6]

                avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2

                # VARIATION 1: Classic Spike - Một tay cao, một tay thấp
                left_very_high = left_wrist['y'] < avg_shoulder_y - height * 0.15
                right_very_high = right_wrist['y'] < avg_shoulder_y - height * 0.15
                left_moderate = avg_shoulder_y - height * 0.15 <= left_wrist['y'] < avg_shoulder_y
                right_moderate = avg_shoulder_y - height * 0.15 <= right_wrist['y'] < avg_shoulder_y

                if (left_very_high and right_moderate) or (right_very_high and left_moderate):
                    confidence = 0.85
                    spike_type = "classic_spike"
                    hitting_hand = "left" if left_very_high else "right"
                    details.append(f"{hitting_hand} hand in classic spike position")

                # VARIATION 2: Two-Hand Spike Preparation - Cả hai tay cao
                elif left_very_high and right_very_high:
                    # Kiểm tra tay nào cao hơn
                    height_diff = abs(left_wrist['y'] - right_wrist['y'])
                    if height_diff > height * 0.08:
                        confidence = 0.8
                        spike_type = "power_spike_prep"
                        higher_hand = "left" if left_wrist['y'] < right_wrist['y'] else "right"
                        details.append(f"Two-hand spike prep, {higher_hand} hand leading")
                    else:
                        confidence = 0.75
                        spike_type = "double_hand_spike"
                        details.append("Both hands equally high for power spike")

                # VARIATION 3: Quick Attack - Tay ngắn, nhanh
                elif (left_moderate and right_moderate):
                    # Kiểm tra góc khuỷu tay (quick attack có góc nhỏ hơn)
                    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    quick_angle = left_elbow_angle < 100 or right_elbow_angle < 100
                    if quick_angle:
                        confidence = 0.7
                        spike_type = "quick_attack"
                        details.append("Compact arm position for quick attack")

                # VARIATION 4: Back Row Attack - Tay cao, cơ thể nghiêng
                if confidence > 0 and 11 in kp_dict:  # Có detect spike + có hip data
                    hip = kp_dict[11]
                    body_lean = abs(avg_shoulder_y - hip['y']) > height * 0.15
                    if body_lean:
                        confidence += 0.1
                        spike_type += "_back_row"
                        details.append("body lean indicates back row attack")

                # VARIATION 5: Cross-Court vs Line Shot - Dựa vào hướng tay
                if confidence > 0:
                    hitting_hand_x = left_wrist['x'] if left_very_high else right_wrist['x']
                    body_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2

                    cross_court = abs(hitting_hand_x - body_center_x) > width * 0.1
                    if cross_court:
                        details.append("cross-court angle detected")
                    else:
                        details.append("straight line attack angle")

            return confidence, spike_type, details

        # 2. SETTING - Cả hai tay nâng lên ở mức vai
        def detect_setting():
            confidence = 0.0
            details = []

            if (9 in kp_dict and 10 in kp_dict and 5 in kp_dict and 6 in kp_dict):
                left_wrist = kp_dict[9]
                right_wrist = kp_dict[10]
                left_shoulder = kp_dict[5]
                right_shoulder = kp_dict[6]

                # Cả hai tay ngang mức vai hoặc hơi cao hơn
                hands_at_shoulder_level = (
                        abs(left_wrist['y'] - left_shoulder['y']) < height * 0.1 and
                        abs(right_wrist['y'] - right_shoulder['y']) < height * 0.1
                )

                # Hai tay gần nhau (setting position)
                hands_close = abs(left_wrist['x'] - right_wrist['x']) < width * 0.15

                if hands_at_shoulder_level and hands_close:
                    confidence = 0.75
                    details.append("Both hands positioned for setting")

            return confidence, details

        # 3. BLOCKING - Variations of blocking positions
        def detect_blocking_variations():
            confidence = 0.0
            details = []
            block_type = "unknown"

            if (9 in kp_dict and 10 in kp_dict and 7 in kp_dict and 8 in kp_dict):
                left_wrist = kp_dict[9]
                right_wrist = kp_dict[10]
                left_elbow = kp_dict[7]
                right_elbow = kp_dict[8]

                # VARIATION 1: Classic Block - Cả hai tay thẳng lên
                left_arm_vertical = abs(left_wrist['x'] - left_elbow['x']) < width * 0.08
                right_arm_vertical = abs(right_wrist['x'] - right_elbow['x']) < width * 0.08

                if left_arm_vertical and right_arm_vertical:
                    confidence = 0.8
                    block_type = "double_block"
                    details.append("Classic double-hand block position")

                # VARIATION 2: Single Block - Một tay chặn
                elif left_arm_vertical or right_arm_vertical:
                    confidence = 0.7
                    block_type = "single_block"
                    blocking_hand = "left" if left_arm_vertical else "right"
                    details.append(f"{blocking_hand} hand single block")

                # VARIATION 3: Soft Block/Tool - Tay hơi nghiêng
                else:
                    # Kiểm tra tay có nâng cao không nhưng không thẳng
                    hands_raised = (left_wrist['y'] < left_elbow['y'] and
                                    right_wrist['y'] < right_elbow['y'])
                    if hands_raised:
                        confidence = 0.6
                        block_type = "soft_block"
                        details.append("Angled hands for soft block or tool")

            return confidence, block_type, details

        # 4. DIGGING/RECEIVE - Tay xuống thấp
        def detect_digging():
            confidence = 0.0
            details = []

            if (9 in kp_dict and 10 in kp_dict and 11 in kp_dict):
                left_wrist = kp_dict[9]
                right_wrist = kp_dict[10]
                hip = kp_dict[11]

                # Cả hai tay thấp hơn hông
                hands_low = (left_wrist['y'] > hip['y'] and right_wrist['y'] > hip['y'])

                # Hai tay gần nhau (platform)
                hands_together = abs(left_wrist['x'] - right_wrist['x']) < width * 0.12

                if hands_low and hands_together:
                    confidence = 0.75
                    details.append("Low platform position for dig/receive")

            return confidence, details

        # Chạy tất cả detections
        spike_conf, spike_type, spike_details = detect_spiking_variations()
        setting_conf, setting_details = detect_setting()
        block_conf, block_type, block_details = detect_blocking_variations()
        dig_conf, dig_details = detect_digging()

        # Ưu tiên theo confidence score
        max_conf = max(spike_conf, setting_conf, block_conf, dig_conf)

        if max_conf == spike_conf and spike_conf > 0.5:
            action_confidence = spike_conf
            detected_actions.append({
                'action': spike_type,
                'confidence': spike_conf,
                'details': '; '.join(spike_details),
                'body_part': 'arms'
            })
        elif max_conf == block_conf and block_conf > 0.5:
            action_confidence = block_conf
            detected_actions.append({
                'action': block_type,
                'confidence': block_conf,
                'details': '; '.join(block_details),
                'body_part': 'arms'
            })
        elif max_conf == setting_conf and setting_conf > 0.5:
            action_confidence = setting_conf
            detected_actions.append({
                'action': 'setting',
                'confidence': setting_conf,
                'details': '; '.join(setting_details),
                'body_part': 'both_arms'
            })
        elif max_conf == dig_conf and dig_conf > 0.5:
            action_confidence = dig_conf
            detected_actions.append({
                'action': 'digging',
                'confidence': dig_conf,
                'details': '; '.join(dig_details),
                'body_part': 'both_arms'
            })

    # ==================== BOXING/MARTIAL ARTS ====================
    elif (any(sport in sport_lower for sport in ['boxing', 'martial arts', 'mma', 'karate', 'taekwondo']) or
          any(sport in allowed_sports for sport in ['boxing', 'martial arts'])):
        print(f"DEBUG BOXING - Entering boxing detection. sport_lower: {sport_lower}, allowed_sports: {allowed_sports}")
        action_confidence = 0.0

        # 1. PUNCHING - Cải thiện detection logic
        def detect_punching_improved():
            confidence = 0.0
            details = []
            punch_type = "unknown"

            if (9 in kp_dict and 10 in kp_dict and 7 in kp_dict and 8 in kp_dict and
                    5 in kp_dict and 6 in kp_dict):

                left_wrist = kp_dict[9]
                right_wrist = kp_dict[10]
                left_elbow = kp_dict[7]
                right_elbow = kp_dict[8]
                left_shoulder = kp_dict[5]
                right_shoulder = kp_dict[6]

                # CẢI THIỆN 1: Giảm ngưỡng extension ratio
                shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
                left_extension = abs(left_wrist['x'] - left_shoulder['x'])
                right_extension = abs(right_wrist['x'] - right_shoulder['x'])

                # GIẢM NGƯỠNG từ 0.8 xuống 0.5
                left_extended = left_extension > shoulder_width * 0.5
                right_extended = right_extension > shoulder_width * 0.5

                # CẢI THIỆN 2: Kiểm tra độ cao tay (boxing stance)
                avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
                left_hand_elevated = left_wrist['y'] < avg_shoulder_y + height * 0.15  # Tay ở mức vai hoặc cao hơn
                right_hand_elevated = right_wrist['y'] < avg_shoulder_y + height * 0.15

                # STRAIGHT PUNCH - Tay duỗi thẳng ra
                if left_extended or right_extended:
                    punching_hand = "left" if left_extended else "right"
                    punch_height = left_wrist['y'] if left_extended else right_wrist['y']
                    shoulder_height = left_shoulder['y'] if left_extended else right_shoulder['y']

                    if abs(punch_height - shoulder_height) < height * 0.15:  # Tăng tolerance
                        punch_type = "straight_punch"
                        confidence = 0.85
                        details.append(f"{punching_hand} straight punch at head level")
                    else:
                        punch_type = "body_shot"
                        confidence = 0.8
                        details.append(f"{punching_hand} body shot")

                # CẢI THIỆN 3: HOOK PUNCH - Giảm yêu cầu góc khuỷu
                elif not (left_extended or right_extended):
                    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    # GIẢM YÊU CẦU từ 60-90 xuống 45-110
                    if 45 <= left_elbow_angle <= 110 and left_hand_elevated:
                        punch_type = "left_hook"
                        confidence = 0.8
                        details.append(f"Left hook with {left_elbow_angle:.1f}° elbow angle")
                    elif 45 <= right_elbow_angle <= 110 and right_hand_elevated:
                        punch_type = "right_hook"
                        confidence = 0.8
                        details.append(f"Right hook with {right_elbow_angle:.1f}° elbow angle")

                # CẢI THIỆN 4: BOXING STANCE DETECTION (mới)
                if confidence == 0.0:
                    # Kiểm tra tư thế boxing cơ bản (cả 2 tay nâng lên)
                    both_hands_up = left_hand_elevated and right_hand_elevated

                    if both_hands_up:
                        # Kiểm tra khoảng cách tay (boxing guard)
                        hand_distance = abs(left_wrist['x'] - right_wrist['x'])

                        if hand_distance > shoulder_width * 0.3:  # Tay rộng ra
                            punch_type = "boxing_stance"
                            confidence = 0.7
                            details.append("Active boxing stance with hands up")
                        else:
                            punch_type = "defensive_guard"
                            confidence = 0.75
                            details.append("Tight defensive guard position")

                # CẢI THIỆN 5: UPPERCUT - Mở rộng detection
                if confidence == 0.0:
                    left_rising = left_wrist['y'] < avg_shoulder_y and left_elbow['y'] > left_wrist['y']
                    right_rising = right_wrist['y'] < avg_shoulder_y and right_elbow['y'] > right_wrist['y']

                    if left_rising or right_rising:
                        punch_type = "uppercut"
                        confidence = 0.75
                        rising_hand = "left" if left_rising else "right"
                        details.append(f"{rising_hand} uppercut motion")



            return confidence, punch_type, details

        # 2. DEFENSIVE GUARD - Cải thiện
        def detect_guard_improved():
            confidence = 0.0
            details = []

            if (9 in kp_dict and 10 in kp_dict and 5 in kp_dict and 6 in kp_dict):
                left_wrist = kp_dict[9]
                right_wrist = kp_dict[10]
                left_shoulder = kp_dict[5]
                right_shoulder = kp_dict[6]

                avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2

                # GIẢM YÊU CẦU - tay ở mức vai hoặc hơi cao hơn
                hands_up = (left_wrist['y'] <= avg_shoulder_y + height * 0.1 and
                            right_wrist['y'] <= avg_shoulder_y + height * 0.1)

                # TĂNG TOLERANCE cho khoảng cách tay
                shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
                hands_close = (abs(left_wrist['x'] - shoulder_center_x) < width * 0.25 and
                               abs(right_wrist['x'] - shoulder_center_x) < width * 0.25)

                if hands_up and hands_close:
                    confidence = 0.8
                    details.append("Defensive guard position with hands up")
                elif hands_up:  # Chỉ cần tay nâng lên
                    confidence = 0.6
                    details.append("Hands elevated in fighting position")

            return confidence, details

        # 3. AGGRESSIVE STANCE - Mới thêm
        def detect_aggressive_stance():
            confidence = 0.0
            details = []

            if (9 in kp_dict and 10 in kp_dict and 5 in kp_dict and 6 in kp_dict and
                    11 in kp_dict and 12 in kp_dict):

                left_wrist = kp_dict[9]
                right_wrist = kp_dict[10]
                left_shoulder = kp_dict[5]
                right_shoulder = kp_dict[6]
                left_hip = kp_dict[11]
                right_hip = kp_dict[12]

                # Kiểm tra tư thế tấn công (1 tay xa, 1 tay gần)
                shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
                left_distance = abs(left_wrist['x'] - shoulder_center_x)
                right_distance = abs(right_wrist['x'] - shoulder_center_x)

                # Một tay xa, một tay gần
                distance_diff = abs(left_distance - right_distance)

                if distance_diff > width * 0.15:  # Chênh lệch đáng kể
                    # Kiểm tra cơ thể có nghiêng về phía trước không
                    avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
                    avg_hip_y = (left_hip['y'] + right_hip['y']) / 2

                    forward_lean = avg_shoulder_y < avg_hip_y - height * 0.05

                    if forward_lean:
                        confidence = 0.75
                        details.append("Aggressive forward stance detected")
                    else:
                        confidence = 0.65
                        details.append("Asymmetric hand position suggesting attack")

            return confidence, details

        # Chạy tất cả detections
        punch_conf, punch_type, punch_details = detect_punching_improved()
        guard_conf, guard_details = detect_guard_improved()
        aggressive_conf, aggressive_details = detect_aggressive_stance()
        # THÊM: Fallback detection cho boxing
        if max(punch_conf, guard_conf, aggressive_conf) == 0:
            print("DEBUG BOXING - Trying fallback detection")

            # Fallback 1: Chỉ cần có tay nâng lên
            if (9 in kp_dict and 10 in kp_dict and 5 in kp_dict and 6 in kp_dict):
                left_wrist = kp_dict[9]
                right_wrist = kp_dict[10]
                left_shoulder = kp_dict[5]
                right_shoulder = kp_dict[6]

                avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2

                # Chỉ cần một tay ngang hoặc cao hơn vai
                hands_elevated = (left_wrist['y'] <= avg_shoulder_y + height * 0.2 or
                                  right_wrist['y'] <= avg_shoulder_y + height * 0.2)

                if hands_elevated:
                    punch_conf = 0.5
                    punch_type = "boxing_stance"
                    punch_details = ["Basic boxing position with elevated hands"]
                    print("DEBUG BOXING - Fallback detection successful")

        # Ưu tiên theo confidence
        max_conf = max(punch_conf, guard_conf, aggressive_conf)

        if max_conf == punch_conf and punch_conf > 0.4:  # Giảm ngưỡng từ 0.5 xuống 0.4
            action_confidence = punch_conf
            detected_actions.append({
                'action': punch_type,
                'confidence': punch_conf,
                'details': '; '.join(punch_details),
                'body_part': 'arms'
            })
        elif max_conf == aggressive_conf and aggressive_conf > 0.4:  # Giảm ngưỡng
            action_confidence = aggressive_conf
            detected_actions.append({
                'action': 'aggressive_stance',
                'confidence': aggressive_conf,
                'details': '; '.join(aggressive_details),
                'body_part': 'full_body'
            })
        elif max_conf == guard_conf and guard_conf > 0.4:  # Giảm ngưỡng
            action_confidence = guard_conf
            detected_actions.append({
                'action': 'defensive_guard',
                'confidence': guard_conf,
                'details': '; '.join(guard_details),
                'body_part': 'both_arms'
            })

        # CẬP NHẬT THÔNG TIN VỀ LOẠI THỂ THAO NẾU PHÁT HIỆN BOXING - LOGIC MỚI
        if action_confidence > 0.5:  # Giảm ngưỡng từ 0.7 xuống 0.5
            detected_action = detected_actions[-1]['action'] if detected_actions else None
            boxing_actions = ['straight_punch', 'left_hook', 'right_hook', 'uppercut', 'body_shot',
                              'defensive_guard', 'boxing_stance', 'aggressive_stance']

            if detected_action in boxing_actions:
                detected_actions[-1]['detected_sport'] = 'boxing'
                # Thêm một flag đặc biệt để force sport type
                detected_actions[-1]['force_boxing'] = True
                print(
                    f"DEBUG - Boxing action detected: {detected_action} - confidence: {detected_actions[-1]['confidence']:.2f} - FORCE BOXING TYPE")


    # ==================== CHẠY/ĐIỀN KINH (RUNNING/TRACK) ====================
    elif (any(sport in sport_lower for sport in ['running', 'track', 'sprint', 'marathon']) and
          any(sport in allowed_sports for sport in ['running', 'track'])):
        action_confidence = 0.0

        # 1. SPRINTING - Chân nâng cao, tay vung mạnh
        if (13 in kp_dict and 14 in kp_dict and 15 in kp_dict and 16 in kp_dict and
                9 in kp_dict and 10 in kp_dict):

            left_knee = kp_dict[13]
            right_knee = kp_dict[14]
            left_ankle = kp_dict[15]
            right_ankle = kp_dict[16]
            left_wrist = kp_dict[9]
            right_wrist = kp_dict[10]

            # Kiểm tra chân nâng cao
            ground_level = height * 0.9
            high_knee = (left_knee['y'] < ground_level * 0.7 or right_knee['y'] < ground_level * 0.7)

            # Kiểm tra tay vung (chênh lệch độ cao giữa hai tay)
            arm_swing = abs(left_wrist['y'] - right_wrist['y']) > height * 0.1

            if high_knee and arm_swing:
                action_confidence = 0.85
                detected_actions.append({
                    'action': 'sprinting',
                    'confidence': action_confidence,
                    'details': 'High knee lift with dynamic arm swing',
                    'body_part': 'full_body'
                })
            elif high_knee:
                action_confidence = 0.7
                detected_actions.append({
                    'action': 'running',
                    'confidence': 0.7,
                    'details': 'Elevated knee position indicating running motion',
                    'body_part': 'legs'
                })

    # ==================== CẦU LÔNG (BADMINTON) ====================
    elif 'badminton' in sport_lower and 'badminton' in allowed_sports:
        action_confidence = 0.0

        # 1. SMASH - Tay nâng cao, chuẩn bị đập mạnh
        def detect_smash():
            confidence = 0.0
            details = []

            if (9 in kp_dict and 10 in kp_dict and 7 in kp_dict and 8 in kp_dict and
                    5 in kp_dict and 6 in kp_dict):

                left_wrist = kp_dict[9]
                right_wrist = kp_dict[10]
                left_elbow = kp_dict[7]
                right_elbow = kp_dict[8]
                left_shoulder = kp_dict[5]
                right_shoulder = kp_dict[6]

                avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2

                # Một tay nâng rất cao (cầm vợt)
                left_very_high = left_wrist['y'] < avg_shoulder_y - height * 0.2
                right_very_high = right_wrist['y'] < avg_shoulder_y - height * 0.2

                if left_very_high or right_very_high:
                    smashing_hand = "left" if left_very_high else "right"

                    # Kiểm tra góc khuỷu tay (smash prep có góc đặc trưng)
                    if smashing_hand == "left":
                        elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    else:
                        elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    if 90 <= elbow_angle <= 150:
                        confidence = 0.85
                        details.append(f"{smashing_hand} hand in smash position (angle: {elbow_angle:.1f}°)")
                    else:
                        confidence = 0.7
                        details.append(f"{smashing_hand} hand raised for overhead shot")

            return confidence, details

        # 2. CLEAR/DROP - Tay vung từ sau ra trước
        def detect_clear_drop():
            confidence = 0.0
            details = []
            shot_type = "unknown"

            if (9 in kp_dict and 10 in kp_dict and 5 in kp_dict and 6 in kp_dict):
                left_wrist = kp_dict[9]
                right_wrist = kp_dict[10]
                left_shoulder = kp_dict[5]
                right_shoulder = kp_dict[6]

                # Kiểm tra tay có duỗi ra không (follow-through)
                shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
                left_extended = abs(left_wrist['x'] - shoulder_center_x) > width * 0.15
                right_extended = abs(right_wrist['x'] - shoulder_center_x) > width * 0.15

                if left_extended or right_extended:
                    hitting_hand = "left" if left_extended else "right"

                    # Xác định shot type dựa trên độ cao
                    if hitting_hand == "left":
                        wrist_height = left_wrist['y']
                        shoulder_height = left_shoulder['y']
                    else:
                        wrist_height = right_wrist['y']
                        shoulder_height = right_shoulder['y']

                    if wrist_height < shoulder_height:
                        shot_type = "clear_shot"
                        confidence = 0.75
                        details.append(f"{hitting_hand} hand clear shot motion")
                    else:
                        shot_type = "drop_shot"
                        confidence = 0.7
                        details.append(f"{hitting_hand} hand drop shot motion")

            return confidence, shot_type, details

        # 3. DEFENSIVE POSITION - Tay thấp, sẵn sàng phòng thủ
        def detect_defensive():
            confidence = 0.0
            details = []

            if (9 in kp_dict and 10 in kp_dict and 11 in kp_dict and 13 in kp_dict and 14 in kp_dict):
                left_wrist = kp_dict[9]
                right_wrist = kp_dict[10]
                hip = kp_dict[11]
                left_knee = kp_dict[13]
                right_knee = kp_dict[14]

                # Tay ở mức thấp (ready position)
                hands_low = (left_wrist['y'] > hip['y'] - height * 0.1 and
                             right_wrist['y'] > hip['y'] - height * 0.1)

                # Chân hơi gấp (athletic stance)
                left_knee_bent = calculate_angle(hip, left_knee, kp_dict[15]) < 160 if 15 in kp_dict else False
                right_knee_bent = calculate_angle(hip, right_knee, kp_dict[16]) < 160 if 16 in kp_dict else False

                if hands_low and (left_knee_bent or right_knee_bent):
                    confidence = 0.7
                    details.append("Low ready position for defensive play")

            return confidence, details

        # Chạy detections
        smash_conf, smash_details = detect_smash()
        clear_conf, clear_type, clear_details = detect_clear_drop()
        def_conf, def_details = detect_defensive()

        # Ưu tiên
        if smash_conf > 0.6:
            action_confidence = smash_conf
            detected_actions.append({
                'action': 'badminton_smash',
                'confidence': smash_conf,
                'details': '; '.join(smash_details),
                'body_part': 'dominant_arm'
            })
        elif clear_conf > 0.6:
            action_confidence = clear_conf
            detected_actions.append({
                'action': clear_type,
                'confidence': clear_conf,
                'details': '; '.join(clear_details),
                'body_part': 'dominant_arm'
            })
        elif def_conf > 0.5:
            action_confidence = def_conf
            detected_actions.append({
                'action': 'defensive_ready',
                'confidence': def_conf,
                'details': '; '.join(def_details),
                'body_part': 'full_body'
            })

    # ==================== GOLF ====================
    elif 'golf' in sport_lower and 'golf' in allowed_sports:
        action_confidence = 0.0

        # 1. BACKSWING - Tay vung lên cao về phía sau
        def detect_backswing():
            confidence = 0.0
            details = []

            if (9 in kp_dict and 10 in kp_dict and 5 in kp_dict and 6 in kp_dict):
                left_wrist = kp_dict[9]
                right_wrist = kp_dict[10]
                left_shoulder = kp_dict[5]
                right_shoulder = kp_dict[6]

                avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
                shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2

                # Cả hai tay nâng cao và hơi lệch về một bên
                hands_high = (left_wrist['y'] < avg_shoulder_y and right_wrist['y'] < avg_shoulder_y)

                # Kiểm tra swing arc (tay lệch sang một bên)
                hands_offset = abs(((left_wrist['x'] + right_wrist['x']) / 2) - shoulder_center_x)
                swing_arc = hands_offset > width * 0.1

                if hands_high and swing_arc:
                    confidence = 0.8
                    swing_side = "right" if ((left_wrist['x'] + right_wrist['x']) / 2) > shoulder_center_x else "left"
                    details.append(f"Backswing motion toward {swing_side} side")

            return confidence, details

        # 2. DOWNSWING/IMPACT - Tay vung xuống
        def detect_downswing():
            confidence = 0.0
            details = []

            if (9 in kp_dict and 10 in kp_dict and 11 in kp_dict):
                left_wrist = kp_dict[9]
                right_wrist = kp_dict[10]
                hip = kp_dict[11]

                # Tay ở mức hông hoặc thấp hơn (impact position)
                hands_at_impact = (left_wrist['y'] >= hip['y'] - height * 0.1 and
                                   right_wrist['y'] >= hip['y'] - height * 0.1)

                # Hai tay gần nhau (grip)
                hands_together = abs(left_wrist['x'] - right_wrist['x']) < width * 0.08

                if hands_at_impact and hands_together:
                    confidence = 0.75
                    details.append("Impact position with hands at ball level")

            return confidence, details

        # 3. FOLLOW THROUGH - Tay vung qua bên kia
        def detect_follow_through():
            confidence = 0.0
            details = []

            if (9 in kp_dict and 10 in kp_dict and 5 in kp_dict and 6 in kp_dict):
                left_wrist = kp_dict[9]
                right_wrist = kp_dict[10]
                left_shoulder = kp_dict[5]
                right_shoulder = kp_dict[6]

                avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
                shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2

                # Tay cao và ở phía đối diện với backswing
                hands_high = (left_wrist['y'] < avg_shoulder_y and right_wrist['y'] < avg_shoulder_y)

                # Kiểm tra follow-through direction
                hands_center_x = (left_wrist['x'] + right_wrist['x']) / 2
                follow_through = abs(hands_center_x - shoulder_center_x) > width * 0.12

                if hands_high and follow_through:
                    confidence = 0.75
                    follow_side = "left" if hands_center_x < shoulder_center_x else "right"
                    details.append(f"Follow-through completion toward {follow_side}")

            return confidence, details

        # 4. PUTTING STANCE - Tay thấp, tư thế cúi
        def detect_putting():
            confidence = 0.0
            details = []

            if (9 in kp_dict and 10 in kp_dict and 5 in kp_dict and 11 in kp_dict):
                left_wrist = kp_dict[9]
                right_wrist = kp_dict[10]
                shoulder = kp_dict[5]
                hip = kp_dict[11]

                # Tay thấp (putting grip)
                hands_low = (left_wrist['y'] > hip['y'] and right_wrist['y'] > hip['y'])

                # Cơ thể cúi về phía trước
                forward_lean = shoulder['y'] < hip['y'] - height * 0.05

                # Hai tay gần nhau và thẳng xuống
                hands_together = abs(left_wrist['x'] - right_wrist['x']) < width * 0.06

                if hands_low and forward_lean and hands_together:
                    confidence = 0.8
                    details.append("Putting stance with forward lean")

            return confidence, details

        # Chạy detections
        back_conf, back_details = detect_backswing()
        down_conf, down_details = detect_downswing()
        follow_conf, follow_details = detect_follow_through()
        putt_conf, putt_details = detect_putting()

        # Ưu tiên theo confidence
        max_conf = max(back_conf, down_conf, follow_conf, putt_conf)

        if max_conf == back_conf and back_conf > 0.6:
            action_confidence = back_conf
            detected_actions.append({
                'action': 'golf_backswing',
                'confidence': back_conf,
                'details': '; '.join(back_details),
                'body_part': 'both_arms'
            })
        elif max_conf == down_conf and down_conf > 0.6:
            action_confidence = down_conf
            detected_actions.append({
                'action': 'golf_impact',
                'confidence': down_conf,
                'details': '; '.join(down_details),
                'body_part': 'both_arms'
            })
        elif max_conf == follow_conf and follow_conf > 0.6:
            action_confidence = follow_conf
            detected_actions.append({
                'action': 'golf_follow_through',
                'confidence': follow_conf,
                'details': '; '.join(follow_details),
                'body_part': 'both_arms'
            })
        elif max_conf == putt_conf and putt_conf > 0.6:
            action_confidence = putt_conf
            detected_actions.append({
                'action': 'putting',
                'confidence': putt_conf,
                'details': '; '.join(putt_details),
                'body_part': 'both_arms'
            })


    # ==================== BƠI LỘI (SWIMMING) ====================
    elif ('swimming' in sport_lower or 'swimming' in allowed_sports or is_swimming_environment):
        print(f"DEBUG SWIMMING - Entering swimming detection")
        print(
            f"sport_lower: {sport_lower}, allowed_sports: {allowed_sports}, is_swimming_environment: {is_swimming_environment}")
        action_confidence = 0.0

        # FREESTYLE STROKE - Một tay duỗi về phía trước
        if (9 in kp_dict and 10 in kp_dict and 5 in kp_dict and 6 in kp_dict):
            left_wrist = kp_dict[9]
            right_wrist = kp_dict[10]
            left_shoulder = kp_dict[5]
            right_shoulder = kp_dict[6]

            # Kiểm tra một tay duỗi xa
            left_extended = abs(left_wrist['x'] - left_shoulder['x']) > width * 0.2
            right_extended = abs(right_wrist['x'] - right_shoulder['x']) > width * 0.2

            if left_extended or right_extended:
                action_confidence = 0.75
                stroke_arm = "left" if left_extended else "right"
                detected_actions.append({
                    'action': 'freestyle_stroke',
                    'confidence': action_confidence,
                    'details': f'{stroke_arm.title()} arm extended for freestyle stroke',
                    'body_part': f'{stroke_arm}_arm'
                })
                print(f"SWIMMING DEBUG - Detected freestyle stroke with {stroke_arm} arm")

        # BREASTSTROKE - Cả hai tay ra ngoài
        if not detected_actions and (9 in kp_dict and 10 in kp_dict and 5 in kp_dict and 6 in kp_dict):
            left_wrist = kp_dict[9]
            right_wrist = kp_dict[10]
            left_shoulder = kp_dict[5]
            right_shoulder = kp_dict[6]

            # Cả hai tay đều ra ngoài (breaststroke)
            left_out = abs(left_wrist['x'] - left_shoulder['x']) > width * 0.15
            right_out = abs(right_wrist['x'] - right_shoulder['x']) > width * 0.15

            if left_out and right_out:
                action_confidence = 0.7
                detected_actions.append({
                    'action': 'breaststroke',
                    'confidence': action_confidence,
                    'details': 'Both arms extended for breaststroke motion',
                    'body_part': 'both_arms'
                })
                print(f"SWIMMING DEBUG - Detected breaststroke")

        # BACKSTROKE - Một tay lên cao
        if not detected_actions and (9 in kp_dict and 10 in kp_dict and 5 in kp_dict and 6 in kp_dict):
            left_wrist = kp_dict[9]
            right_wrist = kp_dict[10]
            left_shoulder = kp_dict[5]
            right_shoulder = kp_dict[6]

            avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2

            # Một tay nâng cao (backstroke)
            left_high = left_wrist['y'] < avg_shoulder_y - height * 0.1
            right_high = right_wrist['y'] < avg_shoulder_y - height * 0.1

            if left_high or right_high:
                action_confidence = 0.65
                stroke_arm = "left" if left_high else "right"
                detected_actions.append({
                    'action': 'backstroke',
                    'confidence': action_confidence,
                    'details': f'{stroke_arm.title()} arm raised for backstroke motion',
                    'body_part': f'{stroke_arm}_arm'
                })
                print(f"SWIMMING DEBUG - Detected backstroke with {stroke_arm} arm")

        # DIVING/STARTING POSITION - Cơ thể cúi về phía trước
        if not detected_actions and (5 in kp_dict and 11 in kp_dict):
            shoulder = kp_dict[5]
            hip = kp_dict[11]

            # Cơ thể nghiêng về phía trước (diving stance)
            forward_lean = shoulder['y'] < hip['y'] - height * 0.05

            if forward_lean:
                action_confidence = 0.6
                detected_actions.append({
                    'action': 'diving_start',
                    'confidence': action_confidence,
                    'details': 'Forward lean indicating diving start position',
                    'body_part': 'full_body'
                })
                print(f"SWIMMING DEBUG - Detected diving start position")

        # FALLBACK cho swimming environment
        if not detected_actions:
            action_confidence = 0.4
            detected_actions.append({
                'action': 'swimming_position',
                'confidence': action_confidence,
                'details': 'General swimming position in pool environment',
                'body_part': 'full_body'
            })
            print(f"SWIMMING DEBUG - Using fallback swimming position")

    # FALLBACK DETECTION - Cho các trường hợp khó detect
    if not detected_actions and len(kp_dict) >= 8:  # Có đủ keypoints nhưng không detect được action
        # Phân tích tổng quát dựa trên body posture
        fallback_confidence = 0.0
        fallback_action = "unknown"
        fallback_details = "General athletic posture detected"

        if 5 in kp_dict and 6 in kp_dict and 11 in kp_dict and 12 in kp_dict:
            shoulders = kp_dict[5], kp_dict[6]
            hips = kp_dict[11], kp_dict[12]

            # Athletic stance detection
            shoulder_width = abs(shoulders[0]['x'] - shoulders[1]['x'])
            hip_width = abs(hips[0]['x'] - hips[1]['x'])
            stance_ratio = shoulder_width / hip_width if hip_width > 0 else 1.0

            if 0.8 <= stance_ratio <= 1.2:  # Balanced athletic stance
                fallback_confidence = 0.4
                fallback_action = "athletic_stance"
                fallback_details = "Balanced athletic position detected"

                # Thêm sport-specific fallback
                if 'soccer' in sport_lower or 'football' in sport_lower:
                    fallback_action = "soccer_ready_position"
                    fallback_details = "Ready position for soccer action"
                elif 'volleyball' in sport_lower:
                    fallback_action = "volleyball_ready_position"
                    fallback_details = "Ready position for volleyball action"

        if fallback_confidence > 0:
            detected_actions.append({
                'action': fallback_action,
                'confidence': fallback_confidence,
                'details': fallback_details,
                'body_part': 'full_body'
            })

    # Tính confidence tổng thể
    if detected_actions:
        overall_confidence = max([action['confidence'] for action in detected_actions])
    else:
        overall_confidence = 0.0

    # Thêm thông tin về body orientation vào kết quả
    for action in detected_actions:
        action['body_orientation'] = body_orientation

    return {
        'detected_actions': detected_actions,
        'confidence': overall_confidence,
        'body_orientation': body_orientation,
        'keypoints_count': len(kp_dict),
        'details': f'Analyzed {len(kp_dict)} keypoints for {sport_type} actions'
    }

def segment_main_subject(img, yolo_seg, main_subject_box):
    results = yolo_seg(img)
    best_mask = None
    best_iou = 0
    for result in results:
        if hasattr(result, 'masks') and result.masks is not None:
            for i, mask in enumerate(result.masks.data):
                box = result.boxes.xyxy[i].cpu().numpy()
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
    analysis["composition_analysis"] if "composition_analysis" in analysis else {}
    depth_map = None
    if 'depth_map' in analysis:
        depth_map = analysis['depth_map']

    env_analysis = analyze_sports_environment(img_data, depth_map)
    result = {
        'sport_type': 'Unknown',
        'framing_quality': 'Unknown',
        'recommended_crop': None,
        'action_focus': 'Unknown'
    }
    sport_equipment = {
        # Đối tượng YOLO cơ bản
        'tennis racket': 'Tennis',
        'boxing gloves': 'Boxing',
        'golf club': 'Golf',
        'badminton racket': 'Badminton',
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

    # ==== MỚI: PHÂN TÍCH ACTION DETECTION ĐỂ SUY RA SPORT TYPE ====
    detected_sport_from_action = None
    action_confidence = 0.0

    # Lấy action detection results từ analysis
    if 'sports_analysis' in analysis and 'action_detection' in analysis['sports_analysis']:
        action_data = analysis['sports_analysis']['action_detection']
        actions = analysis['sports_analysis']['action_detection'].get('detected_actions', [])
        for action in actions:
            print(
                f"DEBUG COMPOSE - Action: {action['action']}, Confidence: {action['confidence']:.2f}, Sport: {action.get('detected_sport', 'unknown')}")
        detected_actions = action_data.get('detected_actions', [])

        if detected_actions:
            # Mapping từ action sang sport type
            action_to_sport = {
                # Boxing/Martial Arts actions
                'straight_punch': 'Boxing',
                'left_hook': 'Boxing',
                'right_hook': 'Boxing',
                'uppercut': 'Boxing',
                'body_shot': 'Boxing',
                'defensive_guard': 'Boxing',
                'high_kick': 'Martial Arts',
                'mid_kick': 'Martial Arts',

                # Badminton actions
                'badminton_smash': 'Badminton',
                'clear_shot': 'Badminton',
                'drop_shot': 'Badminton',
                'defensive_ready': 'Badminton',

                # Golf actions
                'golf_backswing': 'Golf',
                'golf_impact': 'Golf',
                'golf_follow_through': 'Golf',
                'putting': 'Golf',

                # Soccer actions
                'shooting': 'Soccer',
                'pre_kick_stance': 'Soccer',
                'approach_run': 'Soccer',
                'dribbling': 'Soccer',

                # Basketball actions (từ code cũ)
                # 'shooting' đã có ở Soccer, cần phân biệt context

                # Volleyball actions
                'classic_spike': 'Volleyball',
                'power_spike_prep': 'Volleyball',
                'double_hand_spike': 'Volleyball',
                'quick_attack': 'Volleyball',
                'double_block': 'Volleyball',
                'single_block': 'Volleyball',
                'soft_block': 'Volleyball',
                'setting': 'Volleyball',
                'digging': 'Volleyball',

                # Tennis actions
                'serving': 'Tennis',
                'forehand': 'Tennis',
                'backhand': 'Tennis',

                # Running/Track actions
                'sprinting': 'Track and Field',
                'running': 'Running',

                # Swimming actions
                'freestyle_stroke': 'Swimming'
            }

            # Tìm action có confidence cao nhất
            best_action = max(detected_actions, key=lambda x: x['confidence'])
            action_name = best_action['action']

            if action_name in action_to_sport and best_action['confidence'] > 0.6:
                detected_sport_from_action = action_to_sport[action_name]
                action_confidence = best_action['confidence']
                print(
                    f"Sport detected from action: {action_name} -> {detected_sport_from_action} (confidence: {action_confidence:.2f})")

                # XỬ LÝ ĐẶC BIỆT: 'shooting' có thể là Soccer hoặc Basketball
                if action_name == 'shooting':
                    # Kiểm tra context để phân biệt
                    if 'soccer ball' in detections.get('classes', []):
                        detected_sport_from_action = 'Soccer'
                    elif 'basketball' in detections.get('classes', []):
                        detected_sport_from_action = 'Basketball'
                    elif 'sports ball' in detections.get('classes', []):
                        # Dựa vào environment để đoán
                        if env_analysis.get('surface_type') == 'grass':
                            detected_sport_from_action = 'Soccer'
                        elif env_analysis.get('surface_type') == 'court':
                            detected_sport_from_action = 'Basketball'
                        else:
                            detected_sport_from_action = 'Soccer'  # Default
    # PRIORITY ƯU TIÊN MỚI: ACTION -> EQUIPMENT -> ENVIRONMENT
    # (để action detection có độ tin cậy cao được ưu tiên)
    decision_log = []

    # PRIORITY 0: Kiểm tra boxing action từ action detection
    if 'sports_analysis' in analysis and 'action_detection' in analysis['sports_analysis'] and \
            analysis['sports_analysis']['action_detection'].get('detected_actions'):

        # Danh sách cụ thể các hành động boxing
        boxing_actions = ['straight_punch', 'left_hook', 'right_hook', 'uppercut', 'body_shot',
                          'defensive_guard', 'boxing_stance', 'aggressive_stance']

        # Trước tiên, kiểm tra trực tiếp tên hành động
        for action in analysis['sports_analysis']['action_detection']['detected_actions']:
            if action['action'] in boxing_actions and action.get('confidence', 0) > 0.6:
                result['sport_type'] = 'Boxing'
                decision_log.append(f"Boxing action detected: {action['action']} ({action['confidence']:.2f})")
                print(f"DEBUG-FIX: Phát hiện boxing action {action['action']}, gán sport_type=Boxing")
                break

        # Nếu chưa tìm thấy, kiểm tra thông qua trường detected_sport (logic cũ)
        if result['sport_type'] != 'Boxing':
            for action in analysis['sports_analysis']['action_detection']['detected_actions']:
                if action.get('confidence', 0) > 0.7 and action.get('detected_sport') == 'boxing':
                    result['sport_type'] = 'Boxing'
                    decision_log.append(
                        f"High-confidence boxing action detection: {action['action']} ({action['confidence']:.2f})")
                    break

    # PRIORITY 1: Action detection có confidence cao (ưu tiên nhất)
    if detected_sport_from_action and action_confidence > 0.7:
        result['sport_type'] = detected_sport_from_action
        decision_log.append(f"High-confidence action detection: {detected_sport_from_action} ({action_confidence:.2f})")

    # PRIORITY 2: Equipment detection
    elif detected_sport and equipment_confidence > 0.6:
        result['sport_type'] = detected_sport
        decision_log.append(f"Equipment detection: {detected_sport} ({equipment_confidence:.2f})")

    # PRIORITY 3: Action detection với confidence trung bình
    elif detected_sport_from_action and action_confidence > 0.6:
        result['sport_type'] = detected_sport_from_action
        decision_log.append(f"Action detection: {detected_sport_from_action} ({action_confidence:.2f})")

    # PRIORITY 4: Environment detection (đặc biệt cho swimming)
    elif detected_sport_from_env and env_confidence > 0.8:
        result['sport_type'] = detected_sport_from_env
        decision_log.append(f"Environment detection: {detected_sport_from_env} ({env_confidence:.2f})")

    # PRIORITY 5: Action detection với confidence thấp hơn
    elif detected_sport_from_action and action_confidence > 0.5:
        result['sport_type'] = detected_sport_from_action
        decision_log.append(
            f"Medium-confidence action detection: {detected_sport_from_action} ({action_confidence:.2f})")

    # FALLBACKS
    elif detected_sport and equipment_confidence > 0.4:  # Equipment fallback
        result['sport_type'] = detected_sport
        decision_log.append(f"Equipment detection (fallback): {detected_sport} ({equipment_confidence:.2f})")
    elif detected_sport_from_env and env_confidence > 0.5:  # Environment fallback
        result['sport_type'] = detected_sport_from_env
        decision_log.append(f"Environment detection (fallback): {detected_sport_from_env} ({env_confidence:.2f})")
    elif detected_sport_from_action and action_confidence > 0.4:  # Action fallback
        result['sport_type'] = detected_sport_from_action
        decision_log.append(f"Action detection (fallback): {detected_sport_from_action} ({action_confidence:.2f})")
        # Kiểm tra lần cuối xem có action boxing nào không
    if result['sport_type'] == 'Unknown' or result['sport_type'] == 'Running':
        if 'sports_analysis' in analysis and 'action_detection' in analysis['sports_analysis']:
            actions = analysis['sports_analysis']['action_detection'].get('detected_actions', [])

            boxing_actions = ['straight_punch', 'left_hook', 'right_hook', 'uppercut', 'body_shot',
                              'defensive_guard', 'boxing_stance', 'aggressive_stance']

            for action in actions:
                if action['action'] in boxing_actions and action.get('confidence', 0) > 0.5:
                    result['sport_type'] = 'Boxing'
                    decision_log.append(
                        f"FINAL CHECK: Found boxing action {action['action']} - overriding to Boxing")
                    break

    # Kiểm tra lần cuối xem có action boxing nào không
    boxing_actions = ['straight_punch', 'left_hook', 'right_hook', 'uppercut', 'body_shot',
                      'defensive_guard', 'boxing_stance', 'aggressive_stance']

    boxing_detected = False
    if 'sports_analysis' in analysis and 'action_detection' in analysis['sports_analysis']:
        actions = analysis['sports_analysis']['action_detection'].get('detected_actions', [])
        for action in actions:
            if action['action'] in boxing_actions and action.get('confidence', 0) > 0.5:
                result['sport_type'] = 'Boxing'
                decision_log.append(f"Final check: Boxing action found: {action['action']}")
                print(f"DEBUG-FIX: Kiểm tra cuối cùng - phát hiện {action['action']} -> set Boxing")
                boxing_detected = True
                break

    # Chỉ sử dụng Running làm mặc định khi không phát hiện boxing
    if not boxing_detected and result['sport_type'] == 'Unknown':
        result['sport_type'] = 'Running'  # Default cuối cùng
        decision_log.append("Default: Running")

    print(f"Sport type decision: {' -> '.join(decision_log)}")

    # Debug equipment vs action relationship
    if detected_sport:
        print(f"Equipment detected: {detected_sport} (confidence: {equipment_confidence:.2f})")
    if detected_sport_from_action:
        print(f"Action-based sport: {detected_sport_from_action} (confidence: {action_confidence:.2f})")
    if detected_sport_from_env:
        print(f"Environment-based sport: {detected_sport_from_env} (confidence: {env_confidence:.2f})")

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
                # CHỈ kết nối các keypoint có confidence đủ cao
                if kp1_id in keypoints and kp2_id in keypoints:
                    pt1 = keypoints[kp1_id]
                    pt2 = keypoints[kp2_id]

                    # Kiểm tra khoảng cách để tránh vẽ các đường quá dài - GIẢM NGƯỠNG XUỐNG
                    distance = np.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))
                    # Khoảng cách tối đa hợp lý giữa các keypoint (GIẢM XUỐNG 30%)
                    max_distance = width * 0.3  # Giảm từ 50% xuống 30% chiều rộng ảnh

                    # THÊM: Kiểm tra confidence của cả hai điểm trong dict ban đầu
                    kp1_conf = 0
                    kp2_conf = 0
                    for kp in person['keypoints']:
                        if kp['id'] == kp1_id:
                            kp1_conf = kp['confidence']
                        if kp['id'] == kp2_id:
                            kp2_conf = kp['confidence']

                    # CHỈ vẽ khi cả 2 điểm đều có confidence cao
                    if kp1_conf < 0.2 or kp2_conf < 0.2 or distance > max_distance:
                        continue  # Bỏ qua các skeleton khi có điểm kém tin cậy hoặc quá dài

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
                    # HIỂN THỊ ACTION DETECTION
                    if 'action_detection' in sports_analysis and sports_analysis['action_detection'].get(
                            'detected_actions'):
                        actions = sports_analysis['action_detection']['detected_actions']
                        y_offset = 30

                        # Hiển thị tiêu đề
                        cv2.putText(pose_viz, "DETECTED ACTIONS:", (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        y_offset += 25

                        # Hiển thị từng action
                        for i, action in enumerate(actions[:3]):  # Hiển thị tối đa 3 actions
                            action_text = f"{action['action'].upper()}: {action['confidence']:.2f}"
                            color = (0, 255, 255) if action['confidence'] > 0.8 else (0, 255, 0)

                            cv2.putText(pose_viz, action_text, (10, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            y_offset += 20

                            # Hiển thị chi tiết ngắn gọn
                            if len(action['details']) < 50:
                                cv2.putText(pose_viz, action['details'], (10, y_offset),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                                y_offset += 15

                        # Hiển thị body orientation
                        orientation = sports_analysis['action_detection'].get('body_orientation', 'unknown')
                        cv2.putText(pose_viz, f"View: {orientation}", (10, y_offset + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
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

        # Hiển thị logic detection
        if 'action_detection' in sports_analysis and sports_analysis['action_detection'].get('detected_actions'):
            best_action = max(sports_analysis['action_detection']['detected_actions'],
                            key=lambda x: x['confidence'])
            print(f"  -> Influenced by detected action: {best_action['action']} ({best_action['confidence']:.2f})")

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

    # HIỂN THỊ KẾT QUẢ ACTION DETECTION
    if 'action_detection' in sports_analysis and sports_analysis['action_detection'].get('detected_actions'):
        print("\nSports Action Detection:")
        action_data = sports_analysis['action_detection']
        print(f"- Body orientation: {action_data.get('body_orientation', 'unknown')}")
        print(f"- Overall confidence: {action_data.get('confidence', 0):.2f}")
        print(f"- Keypoints analyzed: {action_data.get('keypoints_count', 0)}")

        for i, action in enumerate(action_data['detected_actions']):
            print(f"  {i + 1}. {action['action'].upper()} ({action['confidence']:.2f})")
            print(f"     - Body part: {action['body_part']}")
            print(f"     - Details: {action['details']}")
            print(f"     - View angle: {action['body_orientation']}")
    else:
        print("\nSports Action Detection: No specific actions detected")

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

    # Step 5: Sports composition analysis (QUAN TRỌNG: Phải sau action detection)
    print("Analyzing composition...")
    composition_analysis = analyze_sports_composition(detections, {
        'sports_analysis': sports_analysis,  # Bao gồm action_detection results
        'depth_map': depth_map
    }, img_data)

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

    # Step 6.6: PHÁT HIỆN HÀNH ĐỘNG THỂ THAO VỚI EQUIPMENT FILTERING
    print("Detecting sports actions with equipment filtering...")
    sport_type = composition_analysis.get('sport_type', 'Unknown')

    # Lấy equipment đã detect
    detected_equipment = action_analysis.get('equipment_types', [])

    # Lấy environment sport analysis
    environment_sport = None
    if 'environment_analysis' in composition_analysis and 'sport_probabilities' in composition_analysis[
        'environment_analysis']:
        env_sports = composition_analysis['environment_analysis']['sport_probabilities']
        if env_sports:
            environment_sport = max(env_sports.items(), key=lambda x: x[1])[0]

    action_detection_results = detect_sports_actions(
        pose_results,
        sport_type,
        img_data['resized_array'].shape,
        detected_equipment=detected_equipment,
        environment_sport=environment_sport
    )

    # Cập nhật sports_analysis với action detection
    sports_analysis['action_detection'] = action_detection_results
    print(f"Detected actions: {[action['action'] for action in action_detection_results.get('detected_actions', [])]}")
    if action_detection_results.get('detected_actions'):
        for action in action_detection_results['detected_actions']:
            print(f"  - {action['action']}: {action['confidence']:.2f} ({action['details']})")

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
    sport_type = composition_analysis.get('sport_type', 'Running').lower()
    sport_type_original = composition_analysis.get('sport_type', 'Running')
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

    # ----------------- 6.5. DESCRIBE DETECTED ACTIONS -----------------
    detected_actions = []
    if 'action_detection' in sports_analysis and sports_analysis['action_detection'].get('detected_actions'):
        actions = sports_analysis['action_detection']['detected_actions']
        high_confidence_actions = [a for a in actions if a['confidence'] > 0.7]

        if high_confidence_actions:
            action_names = [action['action'] for action in high_confidence_actions]

            # Tạo mô tả hành động tự nhiên
            action_descriptions = {
                'shooting': ['taking a shot', 'in a shooting motion', 'preparing to shoot'],
                'pre_kick_stance': ['positioning for a kick', 'in pre-kick stance', 'preparing to strike the ball'],
                'approach_run': ['approaching the ball', 'in run-up motion', 'building momentum for the kick'],
                'dribbling': ['dribbling the ball', 'controlling the ball', 'maneuvering with the ball'],
                'serving': ['serving the ball', 'in a serving position', 'executing a serve'],
                'forehand': ['executing a forehand stroke', 'in a forehand swing'],
                'backhand': ['performing a backhand shot', 'in a backhand position'],
                'classic_spike': ['spiking the ball powerfully', 'in classic attack position'],
                'power_spike_prep': ['preparing for a power spike', 'winding up for attack'],
                'double_hand_spike': ['executing a two-handed spike', 'in powerful attack stance'],
                'quick_attack': ['performing a quick attack', 'in rapid strike position'],
                'double_block': ['blocking at the net', 'in defensive block position'],
                'single_block': ['single-hand blocking', 'defending with one arm'],
                'soft_block': ['soft blocking technique', 'angled defensive position'],
                'setting': ['setting up teammates', 'in setting position'],
                'digging': ['digging the ball', 'in defensive receive position'],
                'sprinting': ['in full sprint', 'sprinting at high speed', 'running dynamically'],
                'running': ['running', 'in motion', 'moving forward'],
                # Boxing/Martial Arts
                'straight_punch': ['throwing a straight punch', 'executing a jab', 'in punching stance'],
                'left_hook': ['delivering a left hook', 'swinging a hook punch'],
                'right_hook': ['executing a right hook', 'throwing a power hook'],
                'uppercut': ['launching an uppercut', 'delivering an uppercut punch'],
                'body_shot': ['targeting the body', 'executing a body punch'],
                'defensive_guard': ['in defensive stance', 'maintaining guard position'],
                'high_kick': ['executing a high kick', 'performing a head kick'],
                'mid_kick': ['delivering a body kick', 'executing a mid-level kick'],

                # Badminton
                'badminton_smash': ['smashing the shuttlecock', 'executing a powerful smash'],
                'clear_shot': ['playing a clear shot', 'hitting a defensive clear'],
                'drop_shot': ['executing a drop shot', 'playing a delicate drop'],
                'defensive_ready': ['in defensive ready position', 'prepared for opponent attack'],

                # Golf
                'golf_backswing': ['in backswing motion', 'preparing the swing'],
                'golf_impact': ['at impact with the ball', 'striking the ball'],
                'golf_follow_through': ['completing the swing', 'in follow-through motion'],
                'putting': ['putting on the green', 'lining up the putt'],
                'freestyle_stroke': ['performing a freestyle stroke', 'swimming freestyle']
            }

            for action_name in action_names:
                if action_name in action_descriptions:
                    action_phrase = random.choice(action_descriptions[action_name])
                    detected_actions.append(action_phrase)

            if detected_actions:
                if len(detected_actions) == 1:
                    action_phrases.append(detected_actions[0])
                else:
                    # Kết hợp nhiều hành động
                    action_phrases.append(f"{detected_actions[0]} and {detected_actions[1]}")

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