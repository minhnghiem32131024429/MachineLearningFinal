import sys
import os
import numpy as np
from ML_1 import generate_sports_caption
# Đặt biến môi trường trước khi import các thư viện khác
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Import PyQt trước qt_material
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTabWidget, QSplitter,
                             QGraphicsDropShadowEffect, QProgressBar, QComboBox, QMessageBox,
                             QSizePolicy, QFrame, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QUrl, QDir
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont, QIcon, QFontDatabase

# Bây giờ mới import qt_material
try:
    import qt_material

    QT_MATERIAL_AVAILABLE = True
except ImportError:
    QT_MATERIAL_AVAILABLE = False
    print("Warning: qt_material not found. Install it with: pip install qt-material")

# Thiết lập matplotlib để sử dụng với PyQt
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Nhập file phân tích ảnh
try:
    from ML_1 import analyze_sports_image

    ML_IMPORT_SUCCESS = True
except ImportError:
    ML_IMPORT_SUCCESS = False
    print("Warning: ML_1.py not found or missing analyze_sports_image function")


# Thread để phân tích ảnh không đông cứng UI
class AnalysisThread(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            # Mô phỏng tiến trình xử lý
            for i in range(0, 101, 10):
                self.progress.emit(i)
                self.msleep(100)  # Giả lập xử lý

            # Gọi hàm phân tích từ ML_1.py
            if ML_IMPORT_SUCCESS:
                try:
                    result = analyze_sports_image(self.image_path)
                    self.finished.emit(result)
                except Exception as e:
                    print(f"Error in analyze_sports_image: {str(e)}")
                    # Nếu phân tích thất bại, sử dụng mẫu
                    mock_result = self.generate_mock_result()
                    self.finished.emit(mock_result)
            else:
                # Mẫu kết quả phân tích giả nếu không tìm thấy ML_1.py
                mock_result = self.generate_mock_result()
                self.finished.emit(mock_result)

        except Exception as e:
            self.error.emit(str(e))

    def generate_mock_result(self):
        """Tạo kết quả giả nếu ML_1.py không được tìm thấy"""
        mock_result = {
            'detections': {'athletes': 3, 'classes': ['person', 'person', 'person', 'sports ball'], 'boxes': [],
                           'scores': []},
            'sports_analysis': {
                'player_dispersion': 0.65,
                'key_subjects': [
                    {'class': 'person', 'prominence': 0.92, 'sharpness': 0.85, 'box': [100, 100, 300, 500]},
                    {'class': 'person', 'prominence': 0.78, 'sharpness': 0.75, 'box': [400, 200, 550, 550]},
                    {'class': 'sports ball', 'prominence': 0.65, 'sharpness': 0.9, 'box': [320, 380, 350, 410]}
                ],
                'sharpness_scores': [0.85, 0.75, 0.9]
            },
            'action_analysis': {
                'action_level': 0.82,
                'action_quality': 'High',
                'equipment_types': ['sports ball']
            },
            'composition_analysis': {
                'sport_type': 'Soccer',
                'framing_quality': 'Good',
                'recommended_crop': {'shift_x': 0.05, 'shift_y': -0.02}
            },
            'facial_analysis': {
                'has_faces': True,
                'dominant_emotion': 'happy',
                'emotion_intensity': 0.85,
                'emotional_value': 'High',
                'emotion_scores': {
                    'happy': 0.85,
                    'neutral': 0.08,
                    'sad': 0.02,
                    'angry': 0.01,
                    'surprise': 0.03,
                    'fear': 0.01
                }
            }
        }
        return mock_result


# Biểu đồ cảm xúc dùng Matplotlib
class EmotionChart(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Tạo figure trước khi gọi parent's __init__
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        # Gọi __init__ của parent sau khi fig đã được tạo
        super(EmotionChart, self).__init__(self.fig)

        self.setParent(parent)
        self.fig.tight_layout()

    def update_chart(self, emotions):
        self.axes.clear()
        if not emotions:
            self.axes.text(0.5, 0.5, "No emotion data", ha='center', va='center')
            self.draw()
            return

        # Sắp xếp cảm xúc theo giá trị giảm dần
        emotions_sorted = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0].capitalize() for item in emotions_sorted]
        values = [item[1] for item in emotions_sorted]

        # Màu cho các cảm xúc
        emotion_colors = {
            'happy': '#4CAF50',
            'neutral': '#9E9E9E',
            'sad': '#2196F3',
            'angry': '#F44336',
            'surprise': '#FF9800',
            'fear': '#9C27B0',
            'disgust': '#795548'
        }

        colors = [emotion_colors.get(label.lower(), '#9E9E9E') for label in labels]

        # Tạo biểu đồ thanh ngang
        bars = self.axes.barh(labels, values, color=colors, alpha=0.7)

        # Thêm giá trị lên thanh
        for bar in bars:
            width = bar.get_width()
            self.axes.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                           f'{width:.2f}', ha='left', va='center')

        self.axes.set_title('Emotion Analysis', fontweight='bold')
        self.axes.set_xlabel('Score')
        self.axes.set_xlim(0, 1.05)
        self.axes.grid(axis='x', alpha=0.3, linestyle='--')

        self.draw()


# Tạo widget hiển thị ảnh với scroll và zoom
class ImageDisplayWidget(QWidget):
    def __init__(self, title="Image", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # Tiêu đề
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(self.title_label)

        # Container để giữ ảnh và tự động co giãn
        self.image_container = QWidget()
        self.image_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        container_layout = QVBoxLayout(self.image_container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # Label để chứa ảnh
        self.image_label = QLabel("No image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.05);
            border-radius: 5px;
            padding: 10px;
        """)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        container_layout.addWidget(self.image_label)
        layout.addWidget(self.image_container)

        # Lưu pixmap gốc để có thể resize
        self.original_pixmap = None

        # Kết nối sự kiện resize
        self.image_container.resizeEvent = self.on_container_resize

    def set_image(self, pixmap=None):
        if pixmap:
            self.original_pixmap = pixmap
            # Hiển thị ảnh với kích thước thích hợp
            self.update_image_size()
        else:
            self.original_pixmap = None
            self.image_label.setText("No image available")

    def on_container_resize(self, event):
        # Khi container thay đổi kích thước, cập nhật ảnh
        self.update_image_size()

    def update_image_size(self):
        if self.original_pixmap:
            # Lấy kích thước có sẵn của container
            container_size = self.image_container.size()

            # Co giãn ảnh để vừa với container, giữ nguyên tỷ lệ
            scaled_pixmap = self.original_pixmap.scaled(
                container_size.width() - 20,  # Trừ đi padding
                container_size.height() - 20,  # Trừ đi padding
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.image_label.setPixmap(scaled_pixmap)


# Class chính của ứng dụng
class SportsAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sports Image Analysis Pro")
        self.setMinimumSize(1200, 800)

        # Áp dụng qt-material nếu có sẵn
        if QT_MATERIAL_AVAILABLE:
            try:
                qt_material.apply_stylesheet(self, theme="light_blue.xml")
            except Exception as e:
                print(f"Error applying material theme: {e}")

        # Cấu trúc UI chính
        self.init_ui()

        # Biến để lưu dữ liệu
        self.current_image_path = None
        self.analysis_results = None

    def init_ui(self):
        # Widget & layout chính
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(10)

        # Header với tiêu đề và chọn theme
        header_layout = QHBoxLayout()

        # Logo và tiêu đề
        title_label = QLabel("Sports Image Analysis Pro")
        title_label.setStyleSheet("""
            font-size: 24px; 
            font-weight: bold; 
            color: #2196F3;
            margin: 10px;
        """)
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Chọn theme nếu qt-material khả dụng
        if QT_MATERIAL_AVAILABLE:
            theme_combo = QComboBox()
            theme_combo.addItem("Light Blue (Default)", "light_blue.xml")
            theme_combo.addItem("Dark Blue", "dark_blue.xml")
            theme_combo.addItem("Light Green", "light_green.xml")
            theme_combo.addItem("Dark Green", "dark_green.xml")
            theme_combo.addItem("Light Red", "light_red.xml")
            theme_combo.addItem("Dark Red", "dark_red.xml")
            theme_combo.setFixedWidth(180)
            theme_combo.currentIndexChanged.connect(self.change_theme)
            header_layout.addWidget(QLabel("Theme:"))
            header_layout.addWidget(theme_combo)

        main_layout.addLayout(header_layout)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Content layout với splitter
        self.splitter = QSplitter(Qt.Horizontal)

        # Panel bên trái - tải lên và phân tích
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)

        # Card cho upload
        upload_card = self.create_card_widget()
        upload_layout = QVBoxLayout(upload_card)

        upload_title = QLabel("Upload Image")
        upload_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        upload_layout.addWidget(upload_title)

        # Ảnh xem trước
        self.preview_image = QLabel("No image selected")
        self.preview_image.setAlignment(Qt.AlignCenter)
        self.preview_image.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.05);
            border-radius: 10px;
            padding: 20px;
            font-size: 16px;
            color: #757575;
        """)
        self.preview_image.setMinimumHeight(300)
        self.preview_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        upload_layout.addWidget(self.preview_image)

        # Nút tải lên
        upload_btn_layout = QHBoxLayout()
        self.upload_btn = QPushButton("Browse Images...")
        self.upload_btn.setMinimumHeight(50)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.upload_btn.clicked.connect(self.open_image)
        upload_btn_layout.addWidget(self.upload_btn)
        upload_layout.addLayout(upload_btn_layout)

        left_layout.addWidget(upload_card)

        # Phân tích card
        analyze_card = self.create_card_widget()
        analyze_layout = QVBoxLayout(analyze_card)

        analyze_title = QLabel("Analyze Image")
        analyze_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        analyze_layout.addWidget(analyze_title)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        analyze_layout.addWidget(self.progress_bar)

        # Nút phân tích
        self.analyze_btn = QPushButton("Analyze Image")
        self.analyze_btn.setMinimumHeight(50)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.start_analysis)
        analyze_layout.addWidget(self.analyze_btn)

        left_layout.addWidget(analyze_card)
        left_layout.addStretch()

        # Panel bên phải - kết quả
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 0, 0)

        # Tab widget cho kết quả
        self.tabs = QTabWidget()

        # Tab 1: Ảnh gốc
        self.tab_original = QWidget()
        tab_original_layout = QVBoxLayout(self.tab_original)
        self.original_image_display = ImageDisplayWidget("Original Image")
        tab_original_layout.addWidget(self.original_image_display)
        self.tabs.addTab(self.tab_original, "Original Image")

        # Tab 2: Object Detection
        self.tab_detection = QWidget()
        tab_detection_layout = QVBoxLayout(self.tab_detection)
        self.detection_image_display = ImageDisplayWidget("Object Detection")
        tab_detection_layout.addWidget(self.detection_image_display)
        self.tabs.addTab(self.tab_detection, "Detection")

        # Tab 3: Main Subject
        self.tab_main_subject = QWidget()
        tab_main_subject_layout = QVBoxLayout(self.tab_main_subject)
        self.main_subject_display = ImageDisplayWidget("Main Subject")
        tab_main_subject_layout.addWidget(self.main_subject_display)
        self.tabs.addTab(self.tab_main_subject, "Main Subject")

        # Tab 4: Depth Map
        self.tab_depth = QWidget()
        tab_depth_layout = QVBoxLayout(self.tab_depth)
        self.depth_image_display = ImageDisplayWidget("Depth Map")
        tab_depth_layout.addWidget(self.depth_image_display)
        self.tabs.addTab(self.tab_depth, "Depth Map")

        # Tab 5: Sharpness
        self.tab_sharpness = QWidget()
        tab_sharpness_layout = QVBoxLayout(self.tab_sharpness)
        self.sharpness_image_display = ImageDisplayWidget("Sharpness Heatmap")
        tab_sharpness_layout.addWidget(self.sharpness_image_display)
        self.tabs.addTab(self.tab_sharpness, "Sharpness")

        # Tab 6: Composition
        self.tab_composition = QWidget()
        tab_composition_layout = QVBoxLayout(self.tab_composition)
        self.composition_image_display = ImageDisplayWidget("Composition Analysis")
        tab_composition_layout.addWidget(self.composition_image_display)
        self.tabs.addTab(self.tab_composition, "Composition")

        # Tab 7: Facial Expression
        self.tab_face = QWidget()
        tab_face_layout = QVBoxLayout(self.tab_face)

        face_content = QSplitter(Qt.Horizontal)

        # Ảnh khuôn mặt
        face_left_widget = QWidget()
        face_left_layout = QVBoxLayout(face_left_widget)

        self.face_image = QLabel("No face detected")
        self.face_image.setAlignment(Qt.AlignCenter)
        self.face_image.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.05);
            border-radius: 10px;
            padding: 20px;
            font-size: 16px;
            color: #757575;
        """)
        self.face_image.setMinimumSize(200, 200)
        face_left_layout.addWidget(self.face_image)

        self.emotion_label = QLabel("No emotion detected")
        self.emotion_label.setAlignment(Qt.AlignCenter)
        self.emotion_label.setStyleSheet("font-size: 16px; margin-top: 10px;")
        face_left_layout.addWidget(self.emotion_label)

        face_left_layout.addStretch()

        # Biểu đồ cảm xúc
        face_right_widget = QWidget()
        face_right_layout = QVBoxLayout(face_right_widget)

        self.emotion_chart = EmotionChart(width=5, height=4)
        face_right_layout.addWidget(self.emotion_chart)

        face_content.addWidget(face_left_widget)
        face_content.addWidget(face_right_widget)
        face_content.setSizes([300, 700])

        tab_face_layout.addWidget(face_content)
        self.tabs.addTab(self.tab_face, "Facial Expression")

        # Tab 8: Thống kê
        self.tab_stats = QWidget()
        tab_stats_layout = QVBoxLayout(self.tab_stats)

        self.stats_label = QLabel("Statistics will appear here")
        self.stats_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.stats_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.05);
            border-radius: 10px;
            padding: 20px;
            font-size: 16px;
            color: #757575;
        """)
        self.stats_label.setWordWrap(True)

        # Thêm scroll area cho thống kê
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setWidget(self.stats_label)
        stats_scroll.setStyleSheet("""
            border: none;
        """)

        tab_stats_layout.addWidget(stats_scroll)
        self.tabs.addTab(self.tab_stats, "Statistics")

        right_layout.addWidget(self.tabs)

        # Thêm các panel vào splitter
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(right_panel)
        self.splitter.setSizes([300, 900])

        main_layout.addWidget(self.splitter)

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_card_widget(self):
        """Tạo widget card với shadow effect"""
        card = QWidget()
        card.setObjectName("card")
        card.setStyleSheet("""
            #card {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
            }
        """)

        # Thêm shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 30))
        card.setGraphicsEffect(shadow)

        return card

    def open_image(self):
        """Mở hộp thoại chọn ảnh"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )

        if file_path:
            try:
                # Hiển thị ảnh xem trước
                pixmap = QPixmap(file_path)
                pixmap = pixmap.scaled(self.preview_image.width(), 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.preview_image.setPixmap(pixmap)

                # Hiển thị ảnh gốc trong tab original
                original_pixmap = QPixmap(file_path)
                self.original_image_display.set_image(original_pixmap)

                # Lưu đường dẫn ảnh
                self.current_image_path = file_path

                # Cho phép phân tích
                self.analyze_btn.setEnabled(True)

                filename = os.path.basename(file_path)
                self.statusBar().showMessage(f"Loaded image: {filename}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not open image: {str(e)}")

    def start_analysis(self):
        """Bắt đầu phân tích ảnh trong luồng riêng biệt"""
        if not self.current_image_path:
            return

        # Vô hiệu hóa nút và hiện progress bar
        self.analyze_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.statusBar().showMessage("Analyzing image...")

        # Tạo và bắt đầu thread
        self.analysis_thread = AnalysisThread(self.current_image_path)
        self.analysis_thread.progress.connect(self.update_progress)
        self.analysis_thread.finished.connect(self.analysis_finished)
        self.analysis_thread.error.connect(self.analysis_error)
        self.analysis_thread.start()

    def update_progress(self, value):
        """Cập nhật tiến trình phân tích"""
        self.progress_bar.setValue(value)

    def analysis_finished(self, result):
        """Xử lý khi phân tích hoàn thành"""
        # Lưu kết quả
        self.analysis_results = result

        # Tạo caption cho ảnh
        caption = generate_sports_caption(result)
        print(f"Generated caption: {caption}")

        # Hiển thị caption ở statusBar
        self.statusBar().showMessage(f"Analysis complete. Caption: {caption}")

        # Cập nhật UI với kết quả từng tab
        self.update_detection_tab(result)
        self.update_main_subject_tab(result)
        self.update_depth_tab(result)
        self.update_sharpness_tab(result)
        self.update_composition_tab(result)
        self.update_face_tab(result)
        self.update_stats_tab(result)

        # Kích hoạt lại các nút
        self.analyze_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        self.statusBar().showMessage("Analysis complete")

        # Tự động chuyển đến tab detection sau khi phân tích xong
        self.tabs.setCurrentIndex(1)  # 1 là tab Detection

    def analysis_error(self, error_message):
        """Xử lý lỗi phân tích"""
        QMessageBox.critical(self, "Analysis Error", str(error_message))

        # Kích hoạt lại các nút
        self.analyze_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        self.statusBar().showMessage("Analysis failed")

    def update_detection_tab(self, result):
        """Cập nhật tab object detection"""
        detection_image = self.extract_component_image("detection")
        if detection_image:
            self.detection_image_display.set_image(detection_image)

    def update_main_subject_tab(self, result):
        """Cập nhật tab main subject"""
        main_subject_image = self.extract_component_image("main_subject")
        if main_subject_image:
            self.main_subject_display.set_image(main_subject_image)

    def update_depth_tab(self, result):
        """Cập nhật tab depth map"""
        depth_image = self.extract_component_image("depth")
        if depth_image:
            self.depth_image_display.set_image(depth_image)

    def update_sharpness_tab(self, result):
        """Cập nhật tab sharpness heatmap"""
        sharpness_image = self.extract_component_image("sharpness")
        if sharpness_image:
            self.sharpness_image_display.set_image(sharpness_image)

    def update_composition_tab(self, result):
        """Cập nhật tab composition analysis"""
        composition_image = self.extract_component_image("composition")
        if composition_image:
            self.composition_image_display.set_image(composition_image)

    def extract_component_image(self, component_type):
        """Trích xuất hình ảnh từ kết quả phân tích dựa trên component_type

        Đây là hàm mới - cố gắng tách các phần khác nhau từ ảnh kết quả
        hoặc tìm trong thư mục face_debug
        """
        result_image_path = "sports_analysis_results.png"
        component_image_path = None

        # Nếu có file kết quả riêng biệt cho thành phần cụ thể
        if component_type == "depth":
            component_image_path = "depth_map.png"
        elif component_type == "sharpness":
            component_image_path = "sharpness_heatmap.png"
        elif component_type == "composition":
            component_image_path = "composition_analysis.png"
        elif component_type == "main_subject":
            component_image_path = "main_subject_highlight.png"
        elif component_type == "detection":
            component_image_path = "detections.png"

        # Kiểm tra xem file riêng biệt có tồn tại không
        if component_image_path and os.path.exists(component_image_path):
            try:
                return QPixmap(component_image_path)
            except Exception as e:
                print(f"Error loading {component_image_path}: {str(e)}")

        # Nếu không tìm thấy file cụ thể, kiểm tra trong debug dir
        debug_dirs = ["face_debug", "debug", "."]
        for debug_dir in debug_dirs:
            for file in os.listdir(debug_dir) if os.path.exists(debug_dir) else []:
                if component_type in file.lower():
                    try:
                        return QPixmap(os.path.join(debug_dir, file))
                    except Exception:
                        pass

        # Nếu ảnh riêng biệt không có sẵn, vẫn dùng kết quả gốc
        if os.path.exists(result_image_path):
            try:
                return QPixmap(result_image_path)
            except Exception as e:
                print(f"Error loading result image: {str(e)}")

        return None

    def update_face_tab(self, result):
        """Cập nhật tab biểu cảm khuôn mặt"""
        facial_analysis = result.get('facial_analysis', {})

        if facial_analysis and facial_analysis.get('has_faces', False):
            # Hiển thị ảnh khuôn mặt nếu có
            if 'face_path' in facial_analysis and os.path.exists(facial_analysis['face_path']):
                try:
                    face_pixmap = QPixmap(facial_analysis['face_path'])
                    face_pixmap = face_pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.face_image.setPixmap(face_pixmap)
                except Exception as e:
                    print(f"Error loading face image: {str(e)}")
                    self.face_image.setText(f"Error loading face image: {str(e)}")

            # Hiển thị thông tin cảm xúc
            emotion = facial_analysis.get('dominant_emotion', 'unknown')
            intensity = facial_analysis.get('emotion_intensity', 0)

            # Đặt màu cho emotion dựa vào loại
            emotion_colors = {
                'happy': '#4CAF50',
                'neutral': '#9E9E9E',
                'sad': '#2196F3',
                'angry': '#F44336',
                'surprise': '#FF9800',
                'fear': '#9C27B0',
                'disgust': '#795548'
            }

            color = emotion_colors.get(emotion.lower(), '#9E9E9E')

            self.emotion_label.setText(
                f"<span style='font-size:20px; font-weight:bold; color:{color};'>{emotion.upper()}</span><br>" +
                f"Intensity: {intensity:.2f}<br>" +
                f"Value: {facial_analysis.get('emotional_value', 'Unknown')}")

            # Cập nhật biểu đồ cảm xúc
            if 'emotion_scores' in facial_analysis:
                self.emotion_chart.update_chart(facial_analysis['emotion_scores'])
            elif 'contextual_scores' in facial_analysis:
                self.emotion_chart.update_chart(facial_analysis['contextual_scores'])
        else:
            # Hiển thị lý do không phát hiện được khuôn mặt
            reason = "Không phát hiện được khuôn mặt"

            if facial_analysis and 'debug_info' in facial_analysis:
                if 'reason' in facial_analysis['debug_info']:
                    reason = facial_analysis['debug_info']['reason']

            if facial_analysis and facial_analysis.get('too_low_confidence', False):
                confidence = facial_analysis.get('face_confidence', 0.0)
                self.face_image.setText(f"Không đủ tin cậy để phân tích\nĐộ tin cậy: {confidence:.2f}")
            else:
                self.face_image.setText(f"{reason}")

            self.emotion_label.setText("Không thể phân tích cảm xúc")
            self.emotion_chart.update_chart({})

    def update_stats_tab(self, result):
        """Cập nhật tab thống kê"""
        html = "<style>body {font-size: 16px;} .section {margin-bottom: 20px;} .header {font-weight: bold; font-size: 18px; color: #2196F3; margin-bottom: 10px;} table {border-collapse: collapse;} </style>"

        # Thêm phần caption (thêm vào đây)
        if 'caption' in result:
            html += "<div class='section'>"
            html += "<div class='header'>Image Caption</div>"
            html += f"<p style='font-size: 18px; padding: 15px; background-color: #f0f0f0; border-radius: 10px;'>{result['caption']}</p>"
            html += "</div>"

        # Tiếp tục code hiện tại...
        # Thông tin cơ bản
        html += "<div class='section'>"
        html += "<div class='header'>General Information</div>"

        # [Code còn lại giữ nguyên]

        if 'composition_analysis' in result:
            comp = result['composition_analysis']
            html += f"<p><b>Sport type:</b> {comp.get('sport_type', 'Unknown')}</p>"
            html += f"<p><b>Framing quality:</b> {comp.get('framing_quality', 'Unknown')}</p>"

        html += f"<p><b>Athletes detected:</b> {result['detections'].get('athletes', 0)}</p>"

        if 'action_analysis' in result:
            action = result['action_analysis']
            html += f"<p><b>Action quality:</b> {action.get('action_quality', 'Unknown')} ({action.get('action_level', 0):.2f})</p>"

            eq_types = action.get('equipment_types', [])
            if eq_types:
                html += f"<p><b>Equipment:</b> {', '.join(eq_types)}</p>"
        html += "</div>"

        # Key subjects
        if 'sports_analysis' in result and 'key_subjects' in result['sports_analysis']:
            html += "<div class='section'>"
            html += "<div class='header'>Key Subjects by Prominence</div>"
            html += "<table border='1'>"
            html += "<tr><th>#</th><th>Class</th><th>Prominence</th><th>Sharpness</th></tr>"

            for i, subject in enumerate(result['sports_analysis']['key_subjects']):
                html += f"<tr><td>{i + 1}</td><td>{subject['class']}</td><td>{subject['prominence']:.2f}</td><td>{subject.get('sharpness', 0):.2f}</td></tr>"

            html += "</table>"
            html += "</div>"

        # Facial expression
        if 'facial_analysis' in result and result['facial_analysis'].get('has_faces', False):
            html += "<div class='section'>"
            html += "<div class='header'>Facial Expression Analysis</div>"

            facial = result['facial_analysis']
            html += f"<p><b>Dominant emotion:</b> {facial.get('dominant_emotion', 'unknown')}</p>"

            if 'original_emotion' in facial and facial['original_emotion'] != facial['dominant_emotion']:
                html += f"<p><b>Original emotion:</b> {facial['original_emotion']}</p>"

            html += f"<p><b>Emotion intensity:</b> {facial.get('emotion_intensity', 0):.2f}</p>"
            html += f"<p><b>Emotional value:</b> {facial.get('emotional_value', 'Unknown')}</p>"

            html += "</div>"

        self.stats_label.setText(html)

    def change_theme(self, index):
        """Thay đổi chủ đề của ứng dụng"""
        if QT_MATERIAL_AVAILABLE:
            combo = self.sender()
            theme = combo.itemData(index)
            try:
                qt_material.apply_stylesheet(self, theme=theme)
            except Exception as e:
                print(f"Error changing theme: {e}")

    def resizeEvent(self, event):
        """Xử lý khi cửa sổ chính thay đổi kích thước"""
        # Gọi phương thức gốc của class cha
        super().resizeEvent(event)

        # Kích hoạt cập nhật kích thước cho tất cả các ảnh
        if hasattr(self, 'original_image_display'):
            self.original_image_display.update_image_size()
        if hasattr(self, 'detection_image_display'):
            self.detection_image_display.update_image_size()
        if hasattr(self, 'main_subject_display'):
            self.main_subject_display.update_image_size()
        if hasattr(self, 'depth_image_display'):
            self.depth_image_display.update_image_size()
        if hasattr(self, 'sharpness_image_display'):
            self.sharpness_image_display.update_image_size()
        if hasattr(self, 'composition_image_display'):
            self.composition_image_display.update_image_size()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SportsAnalysisApp()
    window.show()
    sys.exit(app.exec_())