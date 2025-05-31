import sys
import os
from ML_1 import generate_sports_caption, analyze_sports_image, generate_smart_suggestion
ML_IMPORT_SUCCESS = True
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTabWidget, QSplitter,
                             QGraphicsDropShadowEffect, QProgressBar, QComboBox, QMessageBox,
                             QSizePolicy, QFrame, QScrollArea, QGroupBox, QGridLayout,
                             QTableWidget, QTableWidgetItem, QLineEdit)
from PyQt5.QtGui import QPixmap, QColor, QBrush, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
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
            self.progress.emit(10)
            self.msleep(50)

            # Bước 1: Preprocessing
            self.progress.emit(20)
            self.msleep(100)

            # Bước 2: Object Detection
            self.progress.emit(40)
            self.msleep(200)

            # Bước 3: Depth Analysis
            self.progress.emit(60)
            self.msleep(200)

            # Bước 4: Facial & Pose Analysis
            self.progress.emit(80)
            self.msleep(200)

            # Gọi hàm phân tích từ ML_1.py
            result = analyze_sports_image(self.image_path)

            self.progress.emit(100)
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
        self.results = []
        # **THÊM DÒNG NÀY:** Thêm các thuộc tính cho filtering và sorting
        self.all_results = []  # Lưu tất cả kết quả gốc
        self.current_sort = "name_asc"  # Mặc định sort theo tên A-Z
        # **KẾT THÚC THÊM**
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Batch Processing Results")
        header.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)

        # **THÊM BLOCK NÀY:** Controls bar với search và sort
        controls_layout = QHBoxLayout()

        # Search box
        search_label = QLabel("Search:")
        search_label.setStyleSheet("font-weight: bold; margin-right: 5px;")
        controls_layout.addWidget(search_label)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Enter image name to search...")
        self.search_box.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #2196F3;
            }
        """)
        self.search_box.textChanged.connect(self.filter_results)
        controls_layout.addWidget(self.search_box)

        # Sort dropdown
        sort_label = QLabel("Sort by:")
        sort_label.setStyleSheet("font-weight: bold; margin-left: 15px; margin-right: 5px;")
        controls_layout.addWidget(sort_label)

        self.sort_combo = QComboBox()
        self.sort_combo.addItems([
            "Name (A-Z)", "Name (Z-A)",
            "Framing Score (High-Low)", "Framing Score (Low-High)",
            "Action Score (High-Low)", "Action Score (Low-High)",
            "Emotion Intensity (High-Low)", "Emotion Intensity (Low-High)"
        ])
        self.sort_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
                min-width: 150px;
            }
        """)
        self.sort_combo.currentTextChanged.connect(self.sort_results)
        controls_layout.addWidget(self.sort_combo)

        controls_layout.addStretch()

        # Results count
        self.count_label = QLabel("0 results")
        self.count_label.setStyleSheet("color: #666; font-style: italic;")
        controls_layout.addWidget(self.count_label)

        layout.addLayout(controls_layout)
        # **KẾT THÚC THÊM**

        # Scroll area chứa kết quả
        self.scroll_area = QScrollArea()
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setSpacing(10)

        self.scroll_area.setWidget(self.scroll_content)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none;")

        layout.addWidget(self.scroll_area)

    def add_result(self, image_path, caption, result=None, scores=None):
        # **THAY THẾ TOÀN BỘ HÀM NÀY:**
        # Lưu vào danh sách gốc
        result_data = {
            'path': image_path,
            'caption': caption,
            'result': result,
            'scores': scores or {}
        }
        self.all_results.append(result_data)
        self.results = self.all_results.copy()

        # Refresh display
        self.refresh_display()
        # **KẾT THÚC THAY THẾ**

    def recreate_caption(self, result_data):
        """Tạo lại caption cho một kết quả cụ thể"""
        try:
            if result_data['result']:
                # Tạo caption mới
                new_caption = generate_sports_caption(result_data['result'])

                # Cập nhật caption
                result_data['caption'] = new_caption

                # Refresh lại display
                self.refresh_display()

                # Hiển thị thông báo nhỏ trong console
                print(f"Caption recreated for: {os.path.basename(result_data['path'])}")

            else:
                print("Cannot recreate caption: No analysis result available")

        except Exception as e:
            print(f"Error recreating caption: {e}")

    # **THÊM HÀM MỚI:** Tạo result card với nút copy
    def create_result_card(self, result_data):
        """Tạo card cho một kết quả"""
        image_path = result_data['path']
        caption = result_data['caption']
        scores = result_data['scores']

        # Tạo card cho kết quả
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

        # Header với tên file và nút copy
        header_layout = QHBoxLayout()

        filename = os.path.basename(image_path)
        filename_label = QLabel(f"<b>{filename}</b>")
        filename_label.setStyleSheet("font-size: 14px; margin-bottom: 5px;")
        header_layout.addWidget(filename_label)

        header_layout.addStretch()

        # Nút copy caption
        copy_caption_btn = QPushButton("📋 Copy Caption")
        copy_caption_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        copy_caption_btn.clicked.connect(lambda: self.copy_to_clipboard(caption, "Caption"))
        header_layout.addWidget(copy_caption_btn)
        # Nút recreate caption
        recreate_caption_btn = QPushButton("🔄 Recreate")
        recreate_caption_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        recreate_caption_btn.clicked.connect(lambda: self.recreate_caption(result_data))
        header_layout.addWidget(recreate_caption_btn)

        # Nút copy file path
        copy_path_btn = QPushButton("📁 Copy Path")
        copy_path_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        copy_path_btn.clicked.connect(lambda: self.copy_to_clipboard(image_path, "File path"))
        header_layout.addWidget(copy_path_btn)

        info_layout.addLayout(header_layout)

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

        info_layout.addWidget(caption_label)

        # Thêm scores nếu có
        if scores:
            scores_widget = QWidget()
            scores_layout = QGridLayout(scores_widget)
            scores_layout.setContentsMargins(5, 5, 5, 5)
            scores_layout.setSpacing(8)

            scores_widget.setStyleSheet("""
                QWidget {
                    background-color: #e8f5e8; 
                    border-radius: 8px; 
                    padding: 8px;
                }
                QLabel {
                    font-size: 12px;
                    background-color: transparent;
                }
            """)

            row = 0

            # Framing Quality
            if 'framing_quality' in scores:
                quality_label = QLabel(f"<b>Framing:</b> {scores['framing_quality']}")
                scores_layout.addWidget(quality_label, row, 0)

                if 'framing_score' in scores:
                    score_label = QLabel(f"<b>Score:</b> {scores['framing_score']:.3f}")
                    scores_layout.addWidget(score_label, row, 1)
                row += 1

            # Action Quality
            if 'action_quality' in scores:
                action_label = QLabel(f"<b>Action:</b> {scores['action_quality']}")
                scores_layout.addWidget(action_label, row, 0)

                if 'action_level' in scores:
                    level_label = QLabel(f"<b>Level:</b> {scores['action_level']:.2f}")
                    scores_layout.addWidget(level_label, row, 1)
                row += 1

            # Emotion
            if 'emotion' in scores:
                emotion_label = QLabel(f"<b>Emotion:</b> {scores['emotion']}")
                scores_layout.addWidget(emotion_label, row, 0)

                if 'emotion_intensity' in scores and scores['emotion'] != 'No face detected':
                    intensity_label = QLabel(f"<b>Intensity:</b> {scores['emotion_intensity']:.2f}")
                    scores_layout.addWidget(intensity_label, row, 1)
                row += 1

            # Nút copy scores
            copy_scores_layout = QHBoxLayout()
            copy_scores_layout.addStretch()

            copy_scores_btn = QPushButton("📊 Copy Scores")
            copy_scores_btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF9800;
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 3px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #F57C00;
                }
            """)
            scores_text = self.format_scores_for_copy(scores)
            copy_scores_btn.clicked.connect(lambda: self.copy_to_clipboard(scores_text, "Scores"))
            copy_scores_layout.addWidget(copy_scores_btn)

            scores_layout.addLayout(copy_scores_layout, row, 0, 1, 2)

            info_layout.addWidget(scores_widget)

        info_layout.addStretch()

        card_layout.addWidget(image_label)
        card_layout.addLayout(info_layout, 1)

        return result_card

    # **THÊM HÀM MỚI:** Copy to clipboard
    def copy_to_clipboard(self, text, data_type):
        """Copy text to clipboard và hiển thị thông báo"""
        try:
            app = QApplication.instance()
            clipboard = app.clipboard()
            clipboard.setText(text)

            # Hiển thị thông báo nhỏ trong console
            print(f"{data_type} copied to clipboard: {text[:50]}...")

        except Exception as e:
            print(f"Error copying to clipboard: {e}")

    # **THÊM HÀM MỚI:** Format scores để copy
    def format_scores_for_copy(self, scores):
        """Format scores thành text để copy"""
        lines = []
        lines.append("=== ANALYSIS SCORES ===")

        if 'framing_quality' in scores:
            lines.append(f"Framing Quality: {scores['framing_quality']}")
            if 'framing_score' in scores:
                lines.append(f"Framing Score: {scores['framing_score']:.3f}")

        if 'action_quality' in scores:
            lines.append(f"Action Quality: {scores['action_quality']}")
            if 'action_level' in scores:
                lines.append(f"Action Level: {scores['action_level']:.2f}")

        if 'emotion' in scores:
            lines.append(f"Emotion: {scores['emotion']}")
            if 'emotion_intensity' in scores and scores['emotion'] != 'No face detected':
                lines.append(f"Emotion Intensity: {scores['emotion_intensity']:.2f}")

        return "\n".join(lines)

    # **THÊM HÀM MỚI:** Lọc kết quả
    def filter_results(self):
        """Lọc kết quả theo tên file"""
        search_text = self.search_box.text().lower().strip()

        if not search_text:
            self.results = self.all_results.copy()
        else:
            self.results = [
                result for result in self.all_results
                if search_text in os.path.basename(result['path']).lower()
            ]

        self.refresh_display()

    # **THÊM HÀM MỚI:** Sắp xếp kết quả
    def sort_results(self):
        """Sắp xếp kết quả theo lựa chọn"""
        sort_type = self.sort_combo.currentText()

        if sort_type == "Name (A-Z)":
            self.results.sort(key=lambda x: os.path.basename(x['path']).lower())
        elif sort_type == "Name (Z-A)":
            self.results.sort(key=lambda x: os.path.basename(x['path']).lower(), reverse=True)
        elif sort_type == "Framing Score (High-Low)":
            self.results.sort(key=lambda x: x['scores'].get('framing_score', 0), reverse=True)
        elif sort_type == "Framing Score (Low-High)":
            self.results.sort(key=lambda x: x['scores'].get('framing_score', 0))
        elif sort_type == "Action Score (High-Low)":
            self.results.sort(key=lambda x: x['scores'].get('action_level', 0), reverse=True)
        elif sort_type == "Action Score (Low-High)":
            self.results.sort(key=lambda x: x['scores'].get('action_level', 0))
        elif sort_type == "Emotion Intensity (High-Low)":
            self.results.sort(key=lambda x: x['scores'].get('emotion_intensity', 0), reverse=True)
        elif sort_type == "Emotion Intensity (Low-High)":
            self.results.sort(key=lambda x: x['scores'].get('emotion_intensity', 0))

        self.refresh_display()

    # **THÊM HÀM MỚI:** Refresh hiển thị
    def refresh_display(self):
        """Refresh hiển thị với kết quả đã lọc/sắp xếp"""
        # Xóa tất cả widget con
        for i in reversed(range(self.scroll_layout.count())):
            child = self.scroll_layout.itemAt(i).widget()
            if child:
                child.setParent(None)

        # Thêm kết quả mới
        for result_data in self.results:
            result_card = self.create_result_card(result_data)
            self.scroll_layout.addWidget(result_card)

        # Cập nhật số lượng
        self.count_label.setText(f"{len(self.results)} results")

    def clear_results(self):
        # **THAY THẾ TOÀN BỘ HÀM NÀY:**
        """Xóa tất cả kết quả"""
        # Xóa tất cả widget con
        for i in reversed(range(self.scroll_layout.count())):
            child = self.scroll_layout.itemAt(i).widget()
            if child:
                child.setParent(None)

        self.results.clear()
        self.all_results.clear()
        if hasattr(self, 'count_label'):
            self.count_label.setText("0 results")
        if hasattr(self, 'search_box'):
            self.search_box.clear()
        # **KẾT THÚC THAY THẾ**

    def export_results(self):
        # **THAY THẾ TOÀN BỘ HÀM NÀY:**
        if not self.results:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "batch_results.txt", "Text Files (*.txt)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("Batch Processing Results\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Total results: {len(self.results)}\n")
                    f.write(f"Sort order: {self.sort_combo.currentText()}\n")
                    search_text = self.search_box.text().strip()
                    if search_text:
                        f.write(f"Search filter: '{search_text}'\n")
                    f.write("=" * 50 + "\n\n")

                    for i, result in enumerate(self.results, 1):
                        f.write(f"{i}. File: {os.path.basename(result['path'])}\n")
                        f.write(f"   Full path: {result['path']}\n")
                        f.write(f"   Caption: {result['caption']}\n")

                        if result['scores']:
                            f.write(f"   Scores:\n")
                            for key, value in result['scores'].items():
                                f.write(f"     - {key}: {value}\n")
                        f.write("\n")

                QMessageBox.information(self, "Export Complete", f"Results exported to {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Could not export results: {str(e)}")
        # **KẾT THÚC THAY THẾ**


class SportsAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sports Image Analyser")
        self.setMinimumSize(1200, 800)
        self.resize(1920, 1080)
        self.showMaximized()

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
        self.caption_frame = QFrame()
        self.caption_frame.setStyleSheet("""
            QFrame {
                background-color: #f0f8ff;
                border-radius: 8px;
                padding: 10px;
                border-left: 4px solid #2196F3;
                margin-bottom: 15px;
            }
        """)
        caption_layout = QVBoxLayout(self.caption_frame)

        caption_title = QLabel("📝 Image Caption")
        caption_title.setStyleSheet("font-weight: bold; font-size: 16px; color: #2196F3;")
        caption_layout.addWidget(caption_title)

        self.caption_label = QLabel("Caption will appear here after analysis")
        self.caption_label.setWordWrap(True)
        self.caption_label.setStyleSheet("font-size: 14px; padding: 5px;")
        caption_layout.addWidget(self.caption_label)

        tab_stats_layout.addWidget(self.caption_frame)
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

        # Nút recreate all captions
        self.recreate_all_btn = QPushButton("🔄 Recreate All Captions")
        self.recreate_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 5px;
                margin-right: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.recreate_all_btn.clicked.connect(self.recreate_all_batch_captions)
        batch_header_layout.addWidget(self.recreate_all_btn)

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

    def recreate_single_caption(self):
        """Tạo lại caption cho ảnh hiện tại"""
        print("Đang cố gắng tạo lại caption trong chế độ single...")

        if not hasattr(self, 'analysis_results') or not self.analysis_results:
            self.statusBar().showMessage("Không có kết quả phân tích", 3000)
            return

        try:
            # Tạo caption mới
            from ML_1 import generate_sports_caption
            new_caption = generate_sports_caption(self.analysis_results)
            print(f"DEBUG: Caption mới: {new_caption}")

            # Cập nhật trong kết quả phân tích
            if isinstance(self.analysis_results, dict):
                self.analysis_results['caption'] = new_caption
                print("DEBUG: Đã cập nhật caption trong kết quả phân tích")

            # Cập nhật caption label mới nếu có
            if hasattr(self, 'current_caption_label'):
                try:
                    self.current_caption_label.setText(new_caption)
                    print("DEBUG: Đã cập nhật current_caption_label")
                except RuntimeError:
                    print("DEBUG: current_caption_label đã bị xóa, rebuild tab")
                    self.update_stats_tab(self.analysis_results)

            # Cập nhật caption label cũ nếu có và vẫn tồn tại
            if hasattr(self, 'caption_label'):
                try:
                    self.caption_label.setText(new_caption)
                    self.caption_label.setStyleSheet("font-size: 14px; padding: 5px; color: #000000;")
                    print("DEBUG: Đã cập nhật caption_label cũ")
                except RuntimeError:
                    print("DEBUG: caption_label cũ đã bị xóa")

            # Hiển thị thông báo thành công
            self.statusBar().showMessage("Đã tạo lại caption thành công", 3000)

        except Exception as e:
            print(f"DEBUG: Lỗi khi tạo lại caption: {str(e)}")
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"Lỗi khi tạo lại caption: {e}", 3000)

    def recreate_all_batch_captions(self):
        """Tạo lại caption cho tất cả ảnh trong batch"""
        if not self.batch_results.all_results:
            self.statusBar().showMessage("Không có kết quả batch nào để xử lý", 3000)
            return

        # Cập nhật status
        self.statusBar().showMessage("Đang tạo lại caption cho tất cả ảnh...")

        # Vô hiệu hóa nút để tránh nhấn nhiều lần
        self.recreate_all_btn.setEnabled(False)

        try:
            # Tính tổng số ảnh cần xử lý
            total = len(self.batch_results.all_results)
            processed = 0

            # Xử lý từng ảnh
            for result_data in self.batch_results.all_results:
                if result_data['result']:
                    # Tạo caption mới
                    new_caption = generate_sports_caption(result_data['result'])
                    # Cập nhật caption
                    result_data['caption'] = new_caption

                processed += 1
                if processed % 5 == 0 or processed == total:
                    self.statusBar().showMessage(f"Đã tạo lại {processed}/{total} caption...")

            # Cập nhật lại hiển thị
            self.batch_results.refresh_display()
            self.statusBar().showMessage(f"Đã tạo lại thành công {total} caption", 3000)

        except Exception as e:
            self.statusBar().showMessage(f"Lỗi khi tạo lại caption: {e}", 3000)

        # Kích hoạt lại nút
        self.recreate_all_btn.setEnabled(True)

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

    def copy_to_clipboard(self, text):
        """Copy text to clipboard và hiển thị thông báo"""
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        self.statusBar().showMessage("Caption copied to clipboard!", 3000)

    def update_batch_progress(self, current, total):
        """Cập nhật tiến trình batch"""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)
        self.statusBar().showMessage(f"Processing image {current}/{total}...")

    def batch_image_completed(self, image_path, result, caption):
        """Xử lý khi một ảnh trong batch hoàn thành"""
        # Trích xuất thông tin score từ result
        scores = {}
        if result and 'composition_analysis' in result:
            comp = result['composition_analysis']
            scores['framing_quality'] = comp.get('framing_quality', 'Unknown')

            # Lấy framing score chi tiết nếu có
            if 'framing_analysis' in comp and 'overall_score' in comp['framing_analysis']:
                scores['framing_score'] = comp['framing_analysis']['overall_score']

        if result and 'action_analysis' in result:
            action = result['action_analysis']
            scores['action_quality'] = action.get('action_quality', 'Unknown')
            scores['action_level'] = action.get('action_level', 0)

        if result and 'facial_analysis' in result and result['facial_analysis'].get('has_faces', False):
            facial = result['facial_analysis']
            scores['emotion'] = facial.get('dominant_emotion', 'None')
            scores['emotion_intensity'] = facial.get('emotion_intensity', 0)
        else:
            scores['emotion'] = 'No face detected'
            scores['emotion_intensity'] = 0

        self.batch_results.add_result(image_path, caption, result, scores)

    def batch_image_error(self, image_path, error_message):
        """Xử lý lỗi trong batch processing"""
        os.path.basename(image_path) if image_path else "Unknown"
        error_caption = f"Error processing image: {error_message}"
        error_scores = {
            'framing_quality': 'Error',
            'action_quality': 'Error',
            'emotion': 'Error'
        }
        self.batch_results.add_result(image_path, error_caption, None, error_scores)

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
        self.analysis_results = result

        # Tạo caption và lưu vào result
        try:
            caption = generate_sports_caption(result)
            result['caption'] = caption  # Lưu caption vào result
            print(f"Generated caption: {caption}")
            self.statusBar().showMessage(f"Analysis complete. Caption: {caption}")

            # Cập nhật caption label cũ nếu có và vẫn tồn tại
            if hasattr(self, 'caption_label'):
                try:
                    self.caption_label.setText(caption)
                    self.caption_label.setStyleSheet("font-size: 14px; padding: 5px; color: #000000;")
                except RuntimeError:
                    # QLabel đã bị xóa, bỏ qua
                    pass
        except Exception as e:
            print(f"Error generating caption: {e}")
            caption = "Error generating caption"
            result['caption'] = caption

        # Cập nhật các tab khác
        self.update_detection_tab(result)
        self.update_main_subject_tab(result)
        self.update_depth_tab(result)
        self.update_sharpness_tab(result)
        self.update_composition_tab(result)
        self.update_face_tab(result)
        self.update_pose_tab(result)
        self.update_stats_tab(result)  # Gọi cuối cùng để tạo caption label mới

        # Kích hoạt lại các nút
        self.analyze_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Analysis complete")
        self.tabs.setCurrentIndex(8)

    def analysis_error(self, error_message):
        QMessageBox.critical(self, "Analysis Error", str(error_message))
        self.analyze_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Analysis failed")

    def update_detection_tab(self, result):
        detection_image = self.extract_component_image("detection")
        if detection_image:
            self.detection_image_display.set_image(detection_image)

    def update_main_subject_tab(self, result):
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
        """Cập nhật tab thống kê với giao diện widget đẹp hơn"""

        # XÓA AN TOÀN LAYOUT CŨ
        old_layout = self.tab_stats.layout()
        if old_layout:
            # Xóa tất cả widget con trước
            while old_layout.count():
                child = old_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            # Xóa layout
            QWidget().setLayout(old_layout)

        # TẠO SCROLL AREA CHO TOÀN BỘ TAB STATISTICS
        main_layout = QVBoxLayout(self.tab_stats)
        main_layout.setContentsMargins(0, 0, 0, 0)  # Loại bỏ margin của tab chính

        # Tạo scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #c0c0c0;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #a0a0a0;
            }
        """)

        # Tạo widget nội dung chính cho scroll area
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(20, 20, 20, 20)
        scroll_layout.setSpacing(15)

        # Set scroll content vào scroll area
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        # CAPTION DISPLAY - SỬ DỤNG CAPTION ĐÃ TẠO
        caption_group = QGroupBox("📝 Image Caption")
        caption_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 21px;
                border: 2px solid #2196F3;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 15px;
                top: -8px;
                padding: 10px 8px 4px 8px;
                color: #2196F3;
                font-size: 21px;
                font-weight: bold;
                background-color: white;
                border: 1px solid #2196F3;
                border-radius: 4px;
            }
        """)

        caption_layout = QVBoxLayout(caption_group)
        caption_layout.setContentsMargins(15, 20, 15, 15)

        # Tạo header với title và nút copy
        caption_header = QHBoxLayout()
        caption_title = QLabel("Image Caption:")
        caption_title.setStyleSheet("font-weight: bold; color: #2196F3; background: transparent; border: none;")
        caption_header.addWidget(caption_title)
        caption_header.addStretch()

        # Nút copy caption
        copy_button = QPushButton()
        copy_button.setIcon(QIcon.fromTheme("edit-copy"))
        copy_button.setText("Copy")
        copy_button.setToolTip("Copy caption to clipboard")
        copy_button.setStyleSheet("""
            QPushButton {
                background-color: #e3f2fd;
                border: 1px solid #90caf9;
                border-radius: 4px;
                padding: 4px 10px;
                color: #1976d2;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #bbdefb;
            }
        """)

        # SỬ DỤNG CAPTION TỪ KẾT QUẢ PHÂN TÍCH HOẶC ĐÃ TẠO
        current_caption = ""
        if 'caption' in result:
            current_caption = result['caption']
        elif hasattr(self, 'analysis_results') and self.analysis_results and 'caption' in self.analysis_results:
            current_caption = self.analysis_results['caption']
        else:
            # Tạo caption mới nếu chưa có
            try:
                current_caption = generate_sports_caption(result)
                result['caption'] = current_caption  # Lưu lại để dùng sau
            except Exception as e:
                current_caption = f"Error generating caption: {str(e)}"

        copy_button.clicked.connect(lambda: self.copy_to_clipboard(current_caption))
        caption_header.addWidget(copy_button)

        # Nút recreate caption
        recreate_button = QPushButton()
        recreate_button.setIcon(QIcon.fromTheme("view-refresh"))
        recreate_button.setText("Recreate")
        recreate_button.setToolTip("Generate a new caption")
        recreate_button.setStyleSheet("""
            QPushButton {
                background-color: #e8f5e9;
                border: 1px solid #a5d6a7;
                border-radius: 4px;
                padding: 4px 10px;
                color: #388e3c;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c8e6c9;
            }
        """)
        recreate_button.clicked.connect(lambda: self.recreate_single_caption())
        caption_header.addWidget(recreate_button)
        caption_layout.addLayout(caption_header)

        # Caption text trong khung
        self.current_caption_label = QLabel(current_caption)
        self.current_caption_label.setStyleSheet("""
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #2196F3;
            font-size: 14px;
            line-height: 1.5;
            color: #212529;
        """)
        self.current_caption_label.setWordWrap(True)

        caption_layout.addWidget(self.current_caption_label)
        scroll_layout.addWidget(caption_group)

        # 1. Summary GroupBox
        summary_group = QGroupBox("📊 Tổng quan phân tích")
        summary_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 21px;
                border: 2px solid #2196F3;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 15px;
                top: -8px;
                padding: 10px 8px 4px 8px;
                color: #2196F3;
                font-size: 21px;
                font-weight: bold;
                background-color: white;
                border: 1px solid #2196F3;
                border-radius: 4px;
            }
        """)

        summary_layout = QGridLayout(summary_group)
        summary_layout.setContentsMargins(15, 20, 15, 15)
        summary_layout.setSpacing(10)

        # Lấy dữ liệu
        athletes_count = result.get('detections', {}).get('athletes', 0)
        sport_type = result.get('composition_analysis', {}).get('sport_type', 'Unknown')
        framing_quality = result.get('composition_analysis', {}).get('framing_quality', 'Unknown')
        action_quality = result.get('action_analysis', {}).get('action_quality', 'Unknown')
        action_level = result.get('action_analysis', {}).get('action_level', 0)

        # Tạo labels với style đẹp
        def create_info_label(title, value, row, col):
            title_label = QLabel(title)
            title_label.setStyleSheet("font-weight: bold; color: #495057; font-size: 12px;")

            value_label = QLabel(str(value))
            value_label.setStyleSheet("""
                background-color: #f8f9fa;
                padding: 8px 12px;
                border-radius: 6px;
                border-left: 4px solid #4CAF50;
                font-size: 14px;
                color: #212529;
            """)

            summary_layout.addWidget(title_label, row * 2, col)
            summary_layout.addWidget(value_label, row * 2 + 1, col)

        create_info_label("🏃 Number of athletes:", athletes_count, 0, 0)
        create_info_label("⚽ Sport type:", sport_type, 0, 1)
        create_info_label("📷 Framing quality:", framing_quality, 1, 0)
        create_info_label("🎯 Action quality:", f"{action_quality} ({action_level:.2f})", 1, 1)

        # Equipment nếu có
        equipment = result.get('action_analysis', {}).get('equipment_types', [])
        if equipment:
            create_info_label("🏈 Equipment:", ', '.join(equipment), 2, 0)

        scroll_layout.addWidget(summary_group)

        # 2. Object Table GroupBox
        if 'sports_analysis' in result and 'key_subjects' in result['sports_analysis']:
            key_subjects = result['sports_analysis']['key_subjects']
            if key_subjects:
                table_group = QGroupBox("🎯 Subject Analysis Table")
                table_group.setStyleSheet("""
                    QGroupBox {
                        font-weight: bold;
                        font-size: 21px;
                        border: 2px solid #FF9800;
                        border-radius: 8px;
                        margin-top: 15px;
                        padding-top: 15px;
                        background-color: white;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        subcontrol-position: top left;
                        left: 15px;
                        top: -8px;
                        padding: 10px 8px 4px 8px;
                        color: #FF9800;
                        font-size: 21px;
                        font-weight: bold;
                        background-color: white;
                        border: 1px solid #FF9800;
                        border-radius: 4px;
                    }
                """)

                table_layout = QVBoxLayout(table_group)
                table_layout.setContentsMargins(15, 20, 15, 15)

                # Tạo QTableWidget
                table = QTableWidget()
                table.setRowCount(len(key_subjects))
                table.setColumnCount(5)
                table.setHorizontalHeaderLabels(['ID', 'Subject', 'Prominence', 'Sharpness', 'Status'])

                # Style cho table
                table.setStyleSheet("""
                    QTableWidget {
                        gridline-color: #dee2e6;
                        background-color: white;
                        alternate-background-color: #f8f9fa;
                        selection-background-color: #e3f2fd;
                        border: 1px solid #dee2e6;
                        border-radius: 6px;
                    }
                    QTableWidget::item {
                        padding: 8px;
                        border: none;
                    }
                    QHeaderView::section {
                        background-color: #495057;
                        color: white;
                        padding: 10px;
                        border: none;
                        font-weight: bold;
                    }
                """)

                table.setAlternatingRowColors(True)
                table.setSelectionBehavior(QTableWidget.SelectRows)
                table.horizontalHeader().setStretchLastSection(True)

                # Điền dữ liệu vào table
                for i, subject in enumerate(key_subjects):
                    prominence = subject.get('prominence', 0)
                    sharpness = subject.get('sharpness', 0)

                    # ID
                    id_item = QTableWidgetItem(str(i + 1))
                    id_item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(i, 0, id_item)

                    # Loại đối tượng
                    class_item = QTableWidgetItem(subject.get('class', 'Unknown'))
                    class_item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(i, 1, class_item)

                    # Điểm nổi bật
                    prominence_item = QTableWidgetItem(f"{prominence:.3f}")
                    prominence_item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(i, 2, prominence_item)

                    # Độ sắc nét
                    sharpness_item = QTableWidgetItem(f"{sharpness:.3f}")
                    sharpness_item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(i, 3, sharpness_item)

                    # Trạng thái với icon và màu
                    if prominence > 0.7 and sharpness > 0.7:
                        status_text = "✔️ Excellent"
                        status_color = QColor(76, 175, 80, 50)  # Xanh lá nhạt
                    elif prominence > 0.5 and sharpness > 0.5:
                        status_text = "⚠️ Good"
                        status_color = QColor(255, 193, 7, 50)  # Vàng nhạt
                    else:
                        status_text = "❌ Poor"
                        status_color = QColor(244, 67, 54, 50)  # Đỏ nhạt

                    status_item = QTableWidgetItem(status_text)
                    status_item.setTextAlignment(Qt.AlignCenter)
                    status_item.setBackground(QBrush(status_color))
                    table.setItem(i, 4, status_item)

                # Tự động điều chỉnh kích thước cột
                table.resizeColumnsToContents()
                table.setMinimumHeight(150)

                table_layout.addWidget(table)
                scroll_layout.addWidget(table_group)

        # 3. Emotion & Suggestion Layout (chia đôi - GIỮ LAYOUT NGANG)
        emotion_suggestion_layout = QHBoxLayout()
        emotion_suggestion_layout.setSpacing(15)
        emotion_suggestion_layout.setContentsMargins(0, 0, 0, 20)  # Tăng margin bottom

        # 3.1. Emotion Analysis (50%)
        emotion_group = QGroupBox("😊 Facial Expression Analysis")
        emotion_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                border: 2px solid #9C27B0;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: white;
                max-height: 400px;  /* THU NHỎ CHIỀU CAO */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 15px;
                top: -8px;
                padding: 8px 6px 3px 6px;  /* THU NHỎ PADDING */
                color: #9C27B0;
                font-size: 14px;  /* THU NHỎ FONT SIZE */
                font-weight: bold;
                background-color: white;
                border: 1px solid #9C27B0;
                border-radius: 4px;
            }
        """)

        emotion_layout = QVBoxLayout(emotion_group)
        emotion_layout.setContentsMargins(10, 15, 10, 10)
        emotion_layout.setSpacing(5)

        if 'facial_analysis' in result and result['facial_analysis'].get('has_faces', False):
            facial = result['facial_analysis']

            # Container cho emotion info (compact version)
            emotion_container = QWidget()
            emotion_container.setStyleSheet("""
                background: linear-gradient(135deg, #fff3e0, #ffe0b2);
                border-radius: 8px;
                padding: 10px;  /* GIẢM PADDING */
                border-left: 4px solid #FF9800;
            """)

            emotion_info_layout = QVBoxLayout(emotion_container)
            emotion_info_layout.setSpacing(5)
            emotion_info_layout.setContentsMargins(5, 5, 5, 5)

            # Emotion header với icon nhỏ hơn
            emotion_header = QHBoxLayout()

            dominant_emotion = facial.get('dominant_emotion', 'unknown')

            # Emotion name và info
            emotion_info_widget = QWidget()
            emotion_detail_layout = QVBoxLayout(emotion_info_widget)
            emotion_detail_layout.setSpacing(2)
            emotion_detail_layout.setContentsMargins(0, 0, 0, 0)

            emotion_name = QLabel(dominant_emotion.title())
            emotion_name.setStyleSheet("font-size: 14px; font-weight: bold; color: #495057;")

            emotion_subtitle = QLabel("Dominant Emotion")
            emotion_subtitle.setStyleSheet("color: #6c757d; font-size: 11px;")

            emotion_detail_layout.addWidget(emotion_name)
            emotion_detail_layout.addWidget(emotion_subtitle)
            emotion_header.addWidget(emotion_info_widget, 1)
            emotion_header.setSpacing(10)
            emotion_header.setContentsMargins(2, 2, 2, 2)

            emotion_info_layout.addLayout(emotion_header)

            # Metrics compact
            emotion_intensity = facial.get('emotion_intensity', 0)
            emotional_value = facial.get('emotional_value', 'Unknown')

            metrics_layout = QHBoxLayout()
            metrics_layout.setSpacing(5)

            # Intensity
            intensity_widget = QWidget()
            intensity_widget.setStyleSheet("""
                background: white; border-radius: 4px; padding: 4px; border: 1px solid #dee2e6;
            """)
            intensity_layout = QVBoxLayout(intensity_widget)
            intensity_layout.setAlignment(Qt.AlignCenter)
            intensity_layout.setSpacing(0)
            intensity_layout.setContentsMargins(2, 2, 2, 2)

            intensity_value = QLabel(f"{emotion_intensity:.2f}")
            intensity_value.setStyleSheet("font-size: 13px; font-weight: bold; color: #2196F3;")
            intensity_value.setAlignment(Qt.AlignCenter)

            intensity_label = QLabel("Intensity")
            intensity_label.setStyleSheet("font-size: 9px; color: #6c757d;")
            intensity_label.setAlignment(Qt.AlignCenter)

            intensity_layout.addWidget(intensity_value)
            intensity_layout.addWidget(intensity_label)

            # Value
            value_widget = QWidget()
            value_widget.setStyleSheet("""
                background: white; border-radius: 4px; padding: 4px; border: 1px solid #dee2e6;
            """)
            value_layout = QVBoxLayout(value_widget)
            value_layout.setAlignment(Qt.AlignCenter)
            value_layout.setSpacing(0)
            value_layout.setContentsMargins(2, 2, 2, 2)
            value_value = QLabel(emotional_value)
            value_value.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3;")
            value_value.setAlignment(Qt.AlignCenter)
            value_value.setMinimumWidth(80)  # Đảm bảo đủ chiều rộng
            value_value.setTextInteractionFlags(Qt.TextSelectableByMouse)  # Cho phép chọn text

            value_label = QLabel("Value")
            value_label.setStyleSheet("font-size: 9px; color: #6c757d;")
            value_label.setAlignment(Qt.AlignCenter)

            value_layout.addWidget(value_value)
            value_layout.addWidget(value_label)

            metrics_layout.addWidget(intensity_widget)
            metrics_layout.addWidget(value_widget)

            emotion_info_layout.addLayout(metrics_layout)
            emotion_layout.addWidget(emotion_container)

            # Original emotion nếu khác (compact)
            if 'original_emotion' in facial and facial['original_emotion'] != dominant_emotion:
                original_widget = QWidget()
                original_widget.setStyleSheet("""
                    background: rgba(255,193,7,0.1); border-radius: 4px; padding: 4px; border: 1px solid #FFC107;
                """)
                original_layout = QHBoxLayout(original_widget)
                original_layout.setContentsMargins(5, 2, 5, 2)

                original_label = QLabel(f"Original: {facial['original_emotion']}")
                original_label.setStyleSheet("font-weight: bold; color: #495057; font-size: 11px;")

                original_layout.addWidget(original_label)
                emotion_layout.addWidget(original_widget)

        else:
            # No face detected (compact)
            no_face_widget = QWidget()
            no_face_widget.setStyleSheet("""
                background: #f8f9fa; border-radius: 6px; padding: 10px; border: 1px dashed #dee2e6;
            """)

            no_face_layout = QVBoxLayout(no_face_widget)
            no_face_layout.setAlignment(Qt.AlignCenter)
            no_face_layout.setContentsMargins(5, 5, 5, 5)
            no_face_layout.setSpacing(2)

            no_face_icon = QLabel("😔")
            no_face_icon.setStyleSheet("font-size: 24px;")
            no_face_icon.setAlignment(Qt.AlignCenter)

            no_face_text = QLabel("No face detected")
            no_face_text.setStyleSheet("font-size: 12px; color: #6c757d; font-style: italic;")
            no_face_text.setAlignment(Qt.AlignCenter)

            no_face_layout.addWidget(no_face_icon)
            no_face_layout.addWidget(no_face_text)

            emotion_layout.addWidget(no_face_widget)

        # 3.2. Smart Suggestions (50%)
        suggestions_group = QGroupBox("💡 Smart Suggestions")
        suggestions_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: white;
                max-height: 400px;  /* THU NHỎ CHIỀU CAO */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 15px;
                top: -8px;
                padding: 8px 6px 3px 6px;  /* THU NHỎ PADDING */
                color: #4CAF50;
                font-size: 14px;  /* THU NHỎ FONT SIZE */
                font-weight: bold;
                background-color: white;
                border: 1px solid #4CAF50;
                border-radius: 4px;
            }
        """)

        suggestions_layout = QVBoxLayout(suggestions_group)
        suggestions_layout.setContentsMargins(10, 15, 10, 10)
        suggestions_layout.setSpacing(5)

        # Import hàm từ ML_1
        from ML_1 import generate_smart_suggestion

        # Lấy 1 suggestion duy nhất
        smart_suggestion = generate_smart_suggestion(result)

        # Tạo suggestion với icon phù hợp
        if "action" in smart_suggestion.lower():
            suggestion_icon = "🎯"
        elif "framing" in smart_suggestion.lower() or "frame" in smart_suggestion.lower():
            suggestion_icon = "📸"
        elif "sharp" in smart_suggestion.lower() or "focus" in smart_suggestion.lower():
            suggestion_icon = "🔍"
        elif "emotion" in smart_suggestion.lower() or "expression" in smart_suggestion.lower():
            suggestion_icon = "😊"
        elif "excellent" in smart_suggestion.lower() or "great" in smart_suggestion.lower():
            suggestion_icon = "✨"
        else:
            suggestion_icon = "💡"

        suggestions = [{
            'icon': suggestion_icon,
            'title': 'Smart Recommendation',
            'desc': smart_suggestion
        }]

        # Default suggestions if none generated
        if not suggestions:
            suggestions = [
                {'icon': '📷', 'title': 'Great Shot!', 'desc': 'This image shows good sports photography fundamentals.'},
                {'icon': '🏃', 'title': 'Dynamic Content', 'desc': 'The athletic action is well captured in this image.'}
            ]

        # Display suggestions (limit to 1)
        for suggestion in suggestions[:1]:
            suggestion_widget = QWidget()
            suggestion_widget.setStyleSheet("""
                background: #f8f9fa;
                border-radius: 8px;
                padding: 10px;  /* GIẢM PADDING */
                border-left: 4px solid #4CAF50;
                margin: 2px 0px;  /* GIẢM MARGIN */
            """)

            suggestion_layout = QHBoxLayout(suggestion_widget)
            suggestion_layout.setSpacing(5)

            # Content
            content_widget = QWidget()
            content_layout = QVBoxLayout(content_widget)
            content_layout.setSpacing(2)
            content_layout.setContentsMargins(0, 0, 0, 0)
            title_label = QLabel(suggestion['title'])
            title_label.setStyleSheet("font-weight: bold; color: #495057; font-size: 14px;")
            desc_label = QLabel(suggestion['desc'])
            desc_label.setStyleSheet("color: #495057; font-size: 14px; line-height: 1.4;")
            desc_label.setWordWrap(True)
            desc_label.setMinimumHeight(50)  # Đảm bảo có đủ chiều cao
            desc_label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # Cho phép chọn text
            content_layout.addWidget(title_label)
            content_layout.addWidget(desc_label)
            suggestion_layout.addWidget(content_widget, 1)
            suggestion_layout.setContentsMargins(5, 5, 5, 5)
            suggestions_layout.addWidget(suggestion_widget)

        emotion_suggestion_layout.addWidget(emotion_group)
        emotion_suggestion_layout.addWidget(suggestions_group)
        scroll_layout.addLayout(emotion_suggestion_layout)

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
        self.batch_list.clear()
        self.analyze_btn.setEnabled(False)

    def export_batch_results(self):
        self.batch_results.export_results()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SportsAnalysisApp()
    window.show()
    sys.exit(app.exec_())