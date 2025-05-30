import sys
import os
from ML_1 import generate_sports_caption, analyze_sports_image
ML_IMPORT_SUCCESS = True
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTabWidget, QSplitter,
                             QGraphicsDropShadowEffect, QProgressBar, QComboBox, QMessageBox,
                             QSizePolicy, QFrame, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QGridLayout, QButtonGroup, QRadioButton
try:
    import qt_material
    QT_MATERIAL_AVAILABLE = True
except ImportError:
    QT_MATERIAL_AVAILABLE = False
    print("Warning: qt_material not found. Install it with: pip install qt-material")
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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
                self.msleep(100)

            # Gọi hàm phân tích từ ML_1.py
            result = analyze_sports_image(self.image_path)
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class BatchAnalysisThread(QThread):
    progress = pyqtSignal(int, int)  # (current, total)
    image_completed = pyqtSignal(str, dict, str)  # (image_path, result, caption)
    error = pyqtSignal(str, str)  # (image_path, error_message)
    finished = pyqtSignal()

    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths

    def run(self):
        try:
            for i, image_path in enumerate(self.image_paths):
                try:
                    # Phân tích ảnh
                    if ML_IMPORT_SUCCESS:
                        result = analyze_sports_image(image_path)
                    else:
                        result = self.generate_mock_result()

                    # Tạo caption
                    caption = generate_sports_caption(result)

                    # Emit kết quả
                    self.image_completed.emit(image_path, result, caption)

                except Exception as e:
                    self.error.emit(image_path, str(e))

                # Cập nhật progress
                self.progress.emit(i + 1, len(self.image_paths))

            self.finished.emit()

        except Exception as e:
            self.error.emit("", f"Batch processing error: {str(e)}")

    def generate_mock_result(self):
        """Tạo kết quả giả cho batch processing"""
        return {
            'detections': {'athletes': 2, 'classes': ['person', 'person'], 'boxes': [], 'scores': []},
            'sports_analysis': {'player_dispersion': 0.7, 'key_subjects': [], 'sharpness_scores': []},
            'action_analysis': {'action_level': 0.8, 'action_quality': 'High', 'equipment_types': []},
            'composition_analysis': {'sport_type': 'Unknown', 'framing_quality': 'Good'},
            'facial_analysis': {'has_faces': False}
        }


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


class BatchResultWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.results = []

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Batch Processing Results")
        header.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)

        # Scroll area chứa kết quả
        self.scroll_area = QScrollArea()
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setSpacing(10)

        self.scroll_area.setWidget(self.scroll_content)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none;")

        layout.addWidget(self.scroll_area)

    def add_result(self, image_path, caption, result=None):
        # Tạo card cho mỗi kết quả
        result_card = QWidget()
        result_card.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 10px;
                padding: 15px;
                margin: 5px;
                border: 1px solid #ddd;
            }
        """)

        card_layout = QHBoxLayout(result_card)

        # Ảnh thumbnail
        image_label = QLabel()
        try:
            pixmap = QPixmap(image_path)
            thumbnail = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(thumbnail)
        except:
            image_label.setText("Error loading image")

        image_label.setMinimumSize(150, 150)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet("border: 1px solid #ccc; border-radius: 5px;")

        # Thông tin
        info_layout = QVBoxLayout()

        # Tên file
        filename = os.path.basename(image_path)
        filename_label = QLabel(f"<b>{filename}</b>")
        filename_label.setStyleSheet("font-size: 14px; margin-bottom: 5px;")

        # Caption
        caption_label = QLabel(caption)
        caption_label.setWordWrap(True)
        caption_label.setStyleSheet("""
            font-size: 13px; 
            padding: 10px; 
            background-color: #f8f9fa; 
            border-radius: 5px;
            border-left: 3px solid #2196F3;
        """)

        info_layout.addWidget(filename_label)
        info_layout.addWidget(caption_label)
        info_layout.addStretch()

        card_layout.addWidget(image_label)
        card_layout.addLayout(info_layout, 1)

        self.scroll_layout.addWidget(result_card)

        # Lưu kết quả
        self.results.append({
            'path': image_path,
            'caption': caption,
            'result': result
        })

    def clear_results(self):
        # Xóa tất cả widget con
        for i in reversed(range(self.scroll_layout.count())):
            child = self.scroll_layout.itemAt(i).widget()
            if child:
                child.setParent(None)

        self.results.clear()

    def export_results(self):
        if not self.results:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "batch_results.txt", "Text Files (*.txt)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("Batch Processing Results\n")
                    f.write("=" * 50 + "\n\n")

                    for i, result in enumerate(self.results, 1):
                        f.write(f"{i}. File: {os.path.basename(result['path'])}\n")
                        f.write(f"   Caption: {result['caption']}\n\n")

                QMessageBox.information(self, "Export Complete", f"Results exported to {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Could not export results: {str(e)}")


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
        self.batch_images = []

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

        # **THÊM VÀO SAU upload_title:**
        # Mode selection - Chọn chế độ xử lý
        mode_layout = QHBoxLayout()
        self.mode_group = QButtonGroup()

        self.single_mode = QRadioButton("Single Image")
        self.batch_mode = QRadioButton("Batch Processing")
        self.single_mode.setChecked(True)

        self.mode_group.addButton(self.single_mode, 0)
        self.mode_group.addButton(self.batch_mode, 1)
        self.mode_group.buttonClicked.connect(self.mode_changed)

        mode_layout.addWidget(self.single_mode)
        mode_layout.addWidget(self.batch_mode)
        mode_layout.addStretch()

        upload_layout.addLayout(mode_layout)
        # **KẾT THÚC THÊM**

        # Ảnh xem trước (for single mode)
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

        # **THÊM VÀO SAU preview_image:**
        # Batch image list (ẩn ban đầu)
        self.batch_list = QListWidget()
        self.batch_list.setMinimumHeight(300)
        self.batch_list.setVisible(False)
        self.batch_list.setStyleSheet("""
            QListWidget {
                background-color: rgba(0, 0, 0, 0.05);
                border-radius: 10px;
                padding: 10px;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #ddd;
            }
            QListWidget::item:hover {
                background-color: #e3f2fd;
            }
        """)
        upload_layout.addWidget(self.batch_list)
        # **KẾT THÚC THÊM**

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

        # **THÊM VÀO SAU upload_btn:**
        # Clear batch button (ẩn ban đầu)
        self.clear_batch_btn = QPushButton("Clear List")
        self.clear_batch_btn.setMinimumHeight(50)
        self.clear_batch_btn.setVisible(False)
        self.clear_batch_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                background-color: #f44336;
                color: white;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.clear_batch_btn.clicked.connect(self.clear_batch_list)
        upload_btn_layout.addWidget(self.clear_batch_btn)
        # **KẾT THÚC THÊM**

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

        # Tab 8: Pose Estimation
        self.tab_pose = QWidget()
        tab_pose_layout = QVBoxLayout(self.tab_pose)
        self.pose_image_display = ImageDisplayWidget("Pose Estimation")
        tab_pose_layout.addWidget(self.pose_image_display)
        self.tabs.addTab(self.tab_pose, "Pose")

        # Tab 9: Statistics
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

        # **THÊM VÀO SAU tab Statistics:**
        # Tab 10: Batch Results (ẩn ban đầu)
        self.tab_batch = QWidget()
        tab_batch_layout = QVBoxLayout(self.tab_batch)

        # Header cho Batch Results với export button
        batch_header_layout = QHBoxLayout()
        batch_title = QLabel("Batch Processing Results")
        batch_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2196F3;")
        batch_header_layout.addWidget(batch_title)
        batch_header_layout.addStretch()

        self.export_btn = QPushButton("Export Results")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.export_btn.clicked.connect(self.export_batch_results)
        batch_header_layout.addWidget(self.export_btn)

        tab_batch_layout.addLayout(batch_header_layout)

        # Batch results widget
        self.batch_results = BatchResultWidget()
        tab_batch_layout.addWidget(self.batch_results)

        self.batch_tab_index = self.tabs.addTab(self.tab_batch, "Batch Results")
        self.tabs.setTabVisible(self.batch_tab_index, False)  # Ẩn ban đầu
        # **KẾT THÚC THÊM**

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
        if self.batch_mode.isChecked():
            # Batch mode - chọn nhiều ảnh
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "Select Images", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
            )

            if file_paths:
                for file_path in file_paths:
                    # Kiểm tra xem ảnh đã có trong list chưa
                    items = [self.batch_list.item(i).text() for i in range(self.batch_list.count())]
                    filename = os.path.basename(file_path)

                    if file_path not in items:
                        item = QListWidgetItem(file_path)
                        item.setText(filename)
                        item.setToolTip(file_path)
                        self.batch_list.addItem(item)

                # Enable analyze button nếu có ảnh
                self.analyze_btn.setEnabled(self.batch_list.count() > 0)

                self.statusBar().showMessage(f"Added {len(file_paths)} images. Total: {self.batch_list.count()}")
        else:
            # Single mode - chọn một ảnh
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
        """Bắt đầu phân tích ảnh"""
        if self.batch_mode.isChecked():
            # Batch processing
            if self.batch_list.count() == 0:
                return

            # Lấy danh sách đường dẫn
            image_paths = []
            for i in range(self.batch_list.count()):
                item = self.batch_list.item(i)
                image_paths.append(item.toolTip())  # toolTip chứa đường dẫn đầy đủ

            # Clear previous results
            self.batch_results.clear_results()

            # Khởi động batch analysis
            self.analyze_btn.setEnabled(False)
            self.upload_btn.setEnabled(False)
            self.clear_batch_btn.setEnabled(False)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.statusBar().showMessage("Starting batch analysis...")

            # Tạo và bắt đầu batch thread
            self.batch_thread = BatchAnalysisThread(image_paths)
            self.batch_thread.progress.connect(self.update_batch_progress)
            self.batch_thread.image_completed.connect(self.batch_image_completed)
            self.batch_thread.error.connect(self.batch_image_error)
            self.batch_thread.finished.connect(self.batch_analysis_finished)
            self.batch_thread.start()

        else:
            # Single image processing (existing code)
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

    def update_batch_progress(self, current, total):
        """Cập nhật tiến trình batch"""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)
        self.statusBar().showMessage(f"Processing image {current}/{total}...")

    def batch_image_completed(self, image_path, result, caption):
        """Xử lý khi một ảnh trong batch hoàn thành"""
        self.batch_results.add_result(image_path, caption, result)

    def batch_image_error(self, image_path, error_message):
        """Xử lý lỗi trong batch processing"""
        filename = os.path.basename(image_path) if image_path else "Unknown"
        error_caption = f"Error processing image: {error_message}"
        self.batch_results.add_result(image_path, error_caption)

    def batch_analysis_finished(self):
        """Xử lý khi batch analysis hoàn thành"""
        # Kích hoạt lại các nút
        self.analyze_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.clear_batch_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        total_images = self.batch_list.count()
        self.statusBar().showMessage(f"Batch analysis complete. Processed {total_images} images.")

        # Chuyển đến tab batch results
        self.tabs.setCurrentIndex(self.batch_tab_index)

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
        self.update_pose_tab(result)
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

    def update_pose_tab(self, result):
        """Cập nhật tab pose estimation"""
        pose_image = self.extract_component_image("pose")
        if pose_image:
            self.pose_image_display.set_image(pose_image)

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
        elif component_type == "pose":
            component_image_path = "pose_estimation.png"
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
        html = """
        <style>
            body {font-size: 16px;}
            .section {margin-bottom: 20px;}
            .header {font-weight: bold; font-size: 18px; color: #2196F3; margin-bottom: 10px;}
            table {border-collapse: collapse; width: 100%;}
            th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}
            th {background-color: #f2f2f2;}
        </style>
        """

        # Image Caption
        if 'caption' in result:
            html += f"""
            <div class='section'>
                <div class='header'>Image Caption</div>
                <p style='font-size: 18px; padding: 15px; background-color: #f0f0f0; border-radius: 10px;'>
                    {result['caption']}
                </p>
            </div>
            """

        # General Information
        html += "<div class='section'><div class='header'>General Information</div>"

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

        # Key Subjects Table
        if 'sports_analysis' in result and 'key_subjects' in result['sports_analysis']:
            html += """
            <div class='section'>
                <div class='header'>Key Subjects by Prominence</div>
                <table>
                    <tr><th>#</th><th>Class</th><th>Prominence</th><th>Sharpness</th></tr>
            """

            for i, subject in enumerate(result['sports_analysis']['key_subjects']):
                html += f"""
                <tr>
                    <td>{i + 1}</td>
                    <td>{subject['class']}</td>
                    <td>{subject['prominence']:.2f}</td>
                    <td>{subject.get('sharpness', 0):.2f}</td>
                </tr>
                """

            html += "</table></div>"

        # Facial Expression Analysis
        if 'facial_analysis' in result and result['facial_analysis'].get('has_faces', False):
            facial = result['facial_analysis']
            html += f"""
            <div class='section'>
                <div class='header'>Facial Expression Analysis</div>
                <p><b>Dominant emotion:</b> {facial.get('dominant_emotion', 'unknown')}</p>
            """

            if 'original_emotion' in facial and facial['original_emotion'] != facial['dominant_emotion']:
                html += f"<p><b>Original emotion:</b> {facial['original_emotion']}</p>"

            html += f"""
                <p><b>Emotion intensity:</b> {facial.get('emotion_intensity', 0):.2f}</p>
                <p><b>Emotional value:</b> {facial.get('emotional_value', 'Unknown')}</p>
            </div>
            """

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

    def mode_changed(self):
        """Xử lý khi thay đổi mode"""
        is_batch = self.batch_mode.isChecked()

        # Hiện/ẩn UI elements
        self.preview_image.setVisible(not is_batch)
        self.batch_list.setVisible(is_batch)
        self.clear_batch_btn.setVisible(is_batch)

        # Hiện/ẩn tabs
        if is_batch:
            # Ẩn các tab phân tích chi tiết, chỉ hiện batch results
            for i in range(1, self.batch_tab_index):  # Bỏ qua tab Original và Batch Results
                self.tabs.setTabVisible(i, False)
            self.tabs.setTabVisible(self.batch_tab_index, True)
            self.tabs.setCurrentIndex(self.batch_tab_index)
        else:
            # Hiện lại tất cả tabs, ẩn batch results
            for i in range(1, self.batch_tab_index):
                self.tabs.setTabVisible(i, True)
            self.tabs.setTabVisible(self.batch_tab_index, False)
            self.tabs.setCurrentIndex(0)

        # Update button text
        if is_batch:
            self.upload_btn.setText("Add Images...")
            self.analyze_btn.setText("Analyze Batch")
        else:
            self.upload_btn.setText("Browse Images...")
            self.analyze_btn.setText("Analyze Image")

        # Reset selections
        self.current_image_path = None
        self.batch_list.clear()
        self.analyze_btn.setEnabled(False)

    def clear_batch_list(self):
        """Xóa danh sách batch"""
        self.batch_list.clear()
        self.analyze_btn.setEnabled(False)

    def export_batch_results(self):
        """Export kết quả batch"""
        self.batch_results.export_results()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SportsAnalysisApp()
    window.show()
    sys.exit(app.exec_())