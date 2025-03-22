import os
import sys
import time
import warnings
import cv2
import numpy as np
import torch
from PIL import Image, ImageQt
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QColor, QPalette
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QComboBox, QSpinBox, QTableWidget, QTableWidgetItem,
                            QHeaderView, QMessageBox, QProgressBar, QCheckBox,
                            QFrame, QStyleFactory)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class DetectionThread(QThread):
    """Thread class for running object detection in the background"""
    update_frames = pyqtSignal(np.ndarray, np.ndarray, object)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, model, camera_id=0, interval=1, save_images=True):
        super().__init__()
        self.model = model
        self.camera_id = camera_id
        self.interval = interval  # Detection interval in seconds
        self.running = False
        self.cap = None
        self.save_images = save_images
    
    def run(self):
        try:
            # Set camera API (DirectShow for Windows)
            api_preference = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(self.camera_id, api_preference)
            
            # Check if webcam is open
            if not self.cap.isOpened():
                self.error.emit(f"Error: Camera {self.camera_id} could not be opened.")
                return
            
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Create output directory
            if self.save_images:
                result_folder = "detection_results"
                os.makedirs(result_folder, exist_ok=True)
            
            frame_index = 0
            self.running = True
            last_detection_time = time.time()
            
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    # Try reopening the camera
                    time.sleep(1)
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.camera_id, api_preference)
                    if not self.cap.isOpened():
                        self.error.emit("Could not connect to camera")
                        break
                    continue
                
                # BGR -> RGB conversion (for PyQt)
                original_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Perform detection on each frame
                results = self.model(original_frame_rgb.copy())  # Process on a copy
                
                # Visualize detection results
                rendered_frame = np.squeeze(results.render())
                
                # Save frames at specific intervals
                current_time = time.time()
                if self.save_images and (current_time - last_detection_time >= self.interval):
                    frame_index += 1
                    output_file = os.path.join(result_folder, f"detection_{frame_index}.jpg")
                    # RGB -> BGR conversion (for OpenCV)
                    cv2.imwrite(output_file, cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
                    last_detection_time = current_time
                
                # Send both frames to the main window
                self.update_frames.emit(original_frame_rgb, rendered_frame, results.pandas().xyxy[0])
                
                # Short sleep to prevent thread from consuming too much CPU
                time.sleep(0.01)
        
        except Exception as e:
            self.error.emit(f"Error occurred during detection: {str(e)}")
        finally:
            if self.cap is not None:
                self.cap.release()
            self.finished.emit()
    
    def stop(self):
        self.running = False
        self.wait()


class KOSAIComputerVisionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KOSAI Computer Vision Application")
        self.resize(1200, 700)
        
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Application theme and font
        self.setup_theme()
        
        # Load model
        self.load_model()
        
        # Create UI elements
        self.setup_ui()
        
        # Check available cameras
        self.available_cameras = self.check_cameras()
        self.populate_camera_list()
        
        # Detection thread
        self.detection_thread = None
    
    def setup_theme(self):
        """Set up application theme and font"""
        # Main font settings
        app_font = QFont("Segoe UI", 10)
        QApplication.setFont(app_font)
        
        # Main color scheme
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f5f5f7;
                color: #2c2c2c;
            }
            QLabel {
                font-weight: 500;
                color: #333333;
                padding: 2px;
            }
            QPushButton {
                background-color: #4a6fa5;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #5a80b6;
            }
            QPushButton:pressed {
                background-color: #3a5f95;
            }
            QPushButton:disabled {
                background-color: #bbbbbb;
                color: #555555;
            }
            QComboBox, QSpinBox {
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
                min-height: 28px;
            }
            QTableWidget {
                border: 1px solid #dddddd;
                background-color: white;
                gridline-color: #f0f0f0;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #4a6fa5;
                color: white;
                font-weight: bold;
                padding: 6px;
                border: none;
            }
            QTableWidget::item {
                padding: 5px;
                border-bottom: 1px solid #f0f0f0;
            }
            QTableWidget::item:selected {
                background-color: #e0ecff;
                color: #222222;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 3px;
            }
        """)
    
    def load_model(self):
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu', force_reload=True)
            self.model.verbose = False
        except Exception as e:
            QMessageBox.critical(self, "Model Loading Error", 
                                f"Error loading model: {str(e)}")
            sys.exit(1)
    
    def setup_ui(self):
        # Top panel (control panel)
        control_panel = QWidget()
        control_panel.setObjectName("controlPanel")
        control_panel.setStyleSheet("""
            #controlPanel {
                background-color: #ffffff;
                border-radius: 8px;
                margin: 8px;
                padding: 12px;
                border: 1px solid #e0e0e0;
            }
        """)
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(15, 10, 15, 10)
        control_layout.setSpacing(15)
        
        # Image/Webcam selection
        self.btn_load_image = QPushButton("Select Image")
        self.btn_load_image.setIcon(QIcon.fromTheme("document-open"))
        self.btn_load_image.clicked.connect(self.load_image)
        
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(150)
        
        self.btn_start_camera = QPushButton("Start Webcam")
        self.btn_start_camera.setIcon(QIcon.fromTheme("camera-video"))
        self.btn_start_camera.clicked.connect(self.toggle_camera)
        
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 10)
        self.interval_spin.setValue(2)
        self.interval_spin.setSuffix(" sec")
        
        self.save_checkbox = QCheckBox("Save Images")
        self.save_checkbox.setChecked(True)
        
        # Add controls to panel
        action_label = QLabel("Action:")
        action_label.setStyleSheet("font-weight: bold;")
        
        camera_label = QLabel("Camera:")
        camera_label.setStyleSheet("font-weight: bold;")
        
        interval_label = QLabel("Save Interval:")
        interval_label.setStyleSheet("font-weight: bold;")
        
        control_layout.addWidget(action_label)
        control_layout.addWidget(self.btn_load_image)
        control_layout.addWidget(camera_label)
        control_layout.addWidget(self.camera_combo)
        control_layout.addWidget(self.btn_start_camera)
        control_layout.addWidget(interval_label)
        control_layout.addWidget(self.interval_spin)
        control_layout.addWidget(self.save_checkbox)
        control_layout.addStretch()
        
        # Middle panel (images and detection results)
        middle_panel = QWidget()
        middle_layout = QHBoxLayout(middle_panel)
        middle_layout.setSpacing(15)
        
        # Original Camera Image
        original_panel = QFrame()
        original_panel.setObjectName("originalPanel")
        original_panel.setStyleSheet("""
            #originalPanel {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
        """)
        original_layout = QVBoxLayout(original_panel)
        original_layout.setContentsMargins(15, 15, 15, 15)
        
        original_title = QLabel("Original Camera Image")
        original_title.setAlignment(Qt.AlignCenter)
        original_title.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #333333;
            padding-bottom: 8px;
            border-bottom: 2px solid #4a6fa5;
            margin-bottom: 10px;
        """)
        
        self.original_label = QLabel("No image")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 300)
        self.original_label.setStyleSheet("""
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            color: #757575;
            font-style: italic;
        """)
        
        original_layout.addWidget(original_title)
        original_layout.addWidget(self.original_label)
        
        # Object Detection Image
        detection_panel = QFrame()
        detection_panel.setObjectName("detectionPanel")
        detection_panel.setStyleSheet("""
            #detectionPanel {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
        """)
        detection_layout = QVBoxLayout(detection_panel)
        detection_layout.setContentsMargins(15, 15, 15, 15)
        
        detection_title = QLabel("Object Detection Image")
        detection_title.setAlignment(Qt.AlignCenter)
        detection_title.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #333333;
            padding-bottom: 8px;
            border-bottom: 2px solid #4a6fa5;
            margin-bottom: 10px;
        """)
        
        self.detection_label = QLabel("No image")
        self.detection_label.setAlignment(Qt.AlignCenter)
        self.detection_label.setMinimumSize(400, 300)
        self.detection_label.setStyleSheet("""
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            color: #757575;
            font-style: italic;
        """)
        
        detection_layout.addWidget(detection_title)
        detection_layout.addWidget(self.detection_label)
        
        # Results table
        results_panel = QFrame()
        results_panel.setObjectName("resultsPanel")
        results_panel.setStyleSheet("""
            #resultsPanel {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
        """)
        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(15, 15, 15, 15)
        
        results_title = QLabel("Detection Results")
        results_title.setAlignment(Qt.AlignCenter)
        results_title.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #333333;
            padding-bottom: 8px;
            border-bottom: 2px solid #4a6fa5;
            margin-bottom: 10px;
        """)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Object", "Confidence", "x,y", "Width,Height"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        results_layout.addWidget(results_title)
        results_layout.addWidget(self.results_table)
        
        # Arrange middle panel
        middle_layout.addWidget(original_panel, 4)  # 40% width
        middle_layout.addWidget(detection_panel, 4)  # 40% width
        middle_layout.addWidget(results_panel, 3)    # 30% width
        
        # Bottom panel (status info)
        bottom_panel = QFrame()
        bottom_panel.setObjectName("statusPanel")
        bottom_panel.setStyleSheet("""
            #statusPanel {
                background-color: #ffffff;
                border-radius: 8px;
                margin: 8px;
                border: 1px solid #e0e0e0;
                max-height: 50px;
            }
        """)
        bottom_layout = QHBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(15, 5, 15, 5)
        
        status_icon = QLabel()
        status_icon.setPixmap(QIcon.fromTheme("dialog-information").pixmap(16, 16))
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #2c7873; font-weight: 500;")
        
        bottom_layout.addWidget(status_icon)
        bottom_layout.addWidget(self.status_label)
        bottom_layout.addStretch()
        
        # Add panels to main layout
        self.main_layout.addWidget(control_panel)
        self.main_layout.addWidget(middle_panel)
        self.main_layout.addWidget(bottom_panel)
    
    def check_cameras(self):
        """Check available cameras"""
        available_cameras = []
        max_cameras_to_check = 3
        
        for i in range(max_cameras_to_check):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_ANY)
                if cap.isOpened():
                    available_cameras.append(i)
                    cap.release()
            except:
                pass
        
        return available_cameras
    
    def populate_camera_list(self):
        """Populate camera list dropdown"""
        self.camera_combo.clear()
        
        if not self.available_cameras:
            self.camera_combo.addItem("No camera found")
            self.btn_start_camera.setEnabled(False)
        else:
            for cam_id in self.available_cameras:
                self.camera_combo.addItem(f"Camera {cam_id}", cam_id)
            self.btn_start_camera.setEnabled(True)
    
    def load_image(self):
        """Select image file and perform detection"""
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if image_path:
            try:
                # Stop detection if running
                if self.detection_thread and self.detection_thread.isRunning():
                    self.stop_detection()
                
                self.status_label.setText(f"Detecting objects: {os.path.basename(image_path)}")
                QApplication.processEvents()
                
                # Load original image
                original_img = cv2.imread(image_path)
                original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                
                # Perform object detection
                results = self.model(original_rgb.copy())  # Process on a copy
                
                # Visualize results
                rendered_img = np.squeeze(results.render())
                
                # Update results table
                self.update_results_table(results.pandas().xyxy[0])
                
                # Display both images
                self.display_image(original_rgb, self.original_label)
                self.display_image(rendered_img, self.detection_label)
                
                # Save result
                output_path = "result.jpg"
                cv2.imwrite(output_path, cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))
                
                self.status_label.setText(f"Detection completed: {os.path.basename(image_path)}")
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error processing image: {str(e)}")
                self.status_label.setText("Error occurred")
    
    def toggle_camera(self):
        """Start/stop camera detection"""
        if self.detection_thread and self.detection_thread.isRunning():
            # Stop if camera is running
            self.stop_detection()
        else:
            # Start camera
            self.start_detection()
    
    def start_detection(self):
        """Start camera detection"""
        camera_id = self.camera_combo.currentData()
        if camera_id is None:
            QMessageBox.warning(self, "Warning", "Please select a valid camera.")
            return
        
        interval = self.interval_spin.value()
        save_images = self.save_checkbox.isChecked()
        
        # Create detection thread
        self.detection_thread = DetectionThread(
            model=self.model,
            camera_id=camera_id,
            interval=interval,
            save_images=save_images
        )
        
        # Connect signals
        self.detection_thread.update_frames.connect(self.update_frames)
        self.detection_thread.error.connect(self.show_error)
        self.detection_thread.finished.connect(self.on_detection_finished)
        
        # Update UI
        self.btn_start_camera.setText("Stop")
        self.btn_load_image.setEnabled(False)
        self.status_label.setText(f"Detecting with Camera {camera_id}...")
        
        # Start thread
        self.detection_thread.start()
    
    def stop_detection(self):
        """Stop detection"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.status_label.setText("Closing camera...")
            QApplication.processEvents()
            
            self.detection_thread.stop()
            
            # Update UI
            self.btn_start_camera.setText("Start Webcam")
            self.btn_load_image.setEnabled(True)
    
    @pyqtSlot(np.ndarray, np.ndarray, object)
    def update_frames(self, original_frame, detection_frame, results):
        """Update both images and detection results from camera"""
        self.display_image(original_frame, self.original_label)
        self.display_image(detection_frame, self.detection_label)
        
        if results is not None:
            self.update_results_table(results)
    
    @pyqtSlot(str)
    def show_error(self, error_message):
        """Show error messages"""
        QMessageBox.warning(self, "Error", error_message)
        self.status_label.setText(f"Error: {error_message}")
    
    @pyqtSlot()
    def on_detection_finished(self):
        """Called when detection is finished"""
        self.btn_start_camera.setText("Start Webcam")
        self.btn_load_image.setEnabled(True)
        self.status_label.setText("Camera closed")
    
    def display_image(self, img_array, label):
        """Display NumPy array on QLabel"""
        h, w, c = img_array.shape
        bytes_per_line = c * w
        
        # Convert NumPy array to QImage
        q_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert QImage to QPixmap and resize
        pixmap = QPixmap.fromImage(q_img)
        
        # Determine image size
        label_size = label.size()
        scaled_pixmap = pixmap.scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        # Display image
        label.setPixmap(scaled_pixmap)
    
    def update_results_table(self, df):
        """Update detection results table"""
        # Clear table
        self.results_table.setRowCount(0)
        
        if df is None or df.empty:
            return
        
        # Add results to table
        for i, row in df.iterrows():
            table_row = self.results_table.rowCount()
            self.results_table.insertRow(table_row)
            
            # Object name - with colored background
            object_item = QTableWidgetItem(str(row['name']))
            object_item.setBackground(QColor(230, 240, 255))
            object_item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self.results_table.setItem(table_row, 0, object_item)
            
            # Confidence value
            confidence = f"{row['confidence']:.2f}"
            confidence_item = QTableWidgetItem(confidence)
            # Color based on confidence (red->green)
            conf_val = float(confidence)
            if conf_val > 0.7:
                confidence_item.setForeground(QColor(0, 140, 0))
            elif conf_val > 0.5:
                confidence_item.setForeground(QColor(180, 140, 0))
            else:
                confidence_item.setForeground(QColor(200, 0, 0))
            self.results_table.setItem(table_row, 1, confidence_item)
            
            # Position (x, y)
            pos = f"({int(row['xmin'])}, {int(row['ymin'])})"
            self.results_table.setItem(table_row, 2, QTableWidgetItem(pos))
            
            # Size (width, height)
            size = f"{int(row['xmax'] - row['xmin'])}×{int(row['ymax'] - row['ymin'])}"
            self.results_table.setItem(table_row, 3, QTableWidgetItem(size))
    
    def closeEvent(self, event):
        """Called when application is closed"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.detection_thread.wait(1000)  # Wait 1 second
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))  # Modern look
    
    # Start application
    window = KOSAIComputerVisionApp()
    window.show()
    
    sys.exit(app.exec_())