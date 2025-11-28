# -*- coding: utf-8 -*-
import os
# Tắt log TensorFlow để tránh spam console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import sys
sys.stdout.reconfigure(encoding='utf-8')
import tensorflow as tf
import numpy as np
import json
import cv2
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QFileDialog
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer

# ===== Config =====
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5

# Global variables for model and labels (will be loaded lazily)
model = None
class_names = None


def load_model_and_labels():
    """Load model and labels if not already loaded"""
    global model, class_names
    if model is None:
        print("Loading model...")
        try:
            # Thử load bình thường
            model = tf.keras.models.load_model("seafood_model.keras")
        except Exception:
            print("Standard load failed, trying with compile=False...")
            # Nếu lỗi, thử load với compile=False (bỏ qua optimizer state)
            model = tf.keras.models.load_model("seafood_model.keras", compile=False)
            
        print("Model loaded successfully!")
    
    if class_names is None:
        print("Loading labels...")
        with open("labels.json", "r", encoding="utf-8") as f:
            class_names = json.load(f)
        print("Labels loaded successfully!")


def predict_image(img_path):
    # Ensure model is loaded
    load_model_and_labels()
    
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_array = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    preds = model.predict(img_array, verbose=0)
    idx = np.argmax(preds[0])
    conf = float(np.max(preds[0]))

    pred_label = class_names[idx]

    if conf < CONFIDENCE_THRESHOLD:
        pred_show = f"Unknown ({pred_label})"
    else:
        pred_show = pred_label

    return img_rgb, pred_show, conf


def predict_frame(frame):
    """Predict from a camera frame"""
    load_model_and_labels()
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    preds = model.predict(img_array, verbose=0)
    idx = np.argmax(preds[0])
    conf = float(np.max(preds[0]))

    pred_label = class_names[idx]

    if conf < CONFIDENCE_THRESHOLD:
        label = "Unknown"
    else:
        label = pred_label

    return label, conf


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seafood Classifier - PyQt5")
        # Cố định kích thước để tránh lỗi layout bị vỡ/phình to
        self.setFixedSize(1100, 700)

       
        self.header_label = QLabel("Môn: Xử Lý Ảnh Và Thị Giác Máy Tính\n Nhận Diện Sinh Vật Biển - Nhóm 09 ")

        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setFont(QFont("Arial", 35))
        self.header_label.setWordWrap(True) # Cho phép xuống dòng nếu quá dài
        self.header_label.setStyleSheet("color: #222; font-weight: bold; margin-bottom: 20px;")

        # ================= Layout chính =================
        main_layout = QHBoxLayout()
        control_layout = QVBoxLayout()
        image_layout = QVBoxLayout()

        # Hiển thị hình
        self.image_label = QLabel("No Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(500, 500)
        self.image_label.setStyleSheet("border: 3px dashed gray;")

        # Hiển thị kết quả
        self.result_label = QLabel("Result: None")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 20))
        self.result_label.setStyleSheet("color: #222; font-weight: bold;")

        # Nút mở ảnh
        self.btn_open = QPushButton("📁 Open Image")
        self.btn_open.setFixedHeight(55)
        self.btn_open.setFont(QFont("Arial", 16))
        self.btn_open.clicked.connect(self.open_img)

        # Nút dự đoán
        self.btn_predict = QPushButton("Predict")
        self.btn_predict.setFixedHeight(55)
        self.btn_predict.setFont(QFont("Arial", 16))
        self.btn_predict.clicked.connect(self.predict)

        # Nút mở camera
        self.btn_camera = QPushButton("Mở Camera")
        self.btn_camera.setFixedHeight(55)
        self.btn_camera.setFont(QFont("Arial", 16))
        self.btn_camera.clicked.connect(self.toggle_camera)

        # Nút thoát
        self.btn_exit = QPushButton("Thoát")
        self.btn_exit.setFixedHeight(55)
        self.btn_exit.setFont(QFont("Arial", 16))
        self.btn_exit.setStyleSheet("background-color: #d32f2f;")
        self.btn_exit.clicked.connect(self.exit_app)

        control_layout.addSpacing(60) 

        control_layout.addWidget(self.btn_open)
        control_layout.addWidget(self.btn_predict)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.btn_camera)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.btn_exit)

        control_layout.addStretch()
        control_layout.addWidget(self.result_label)
        control_layout.addSpacing(50)

        # Layout hình
        image_layout.addWidget(self.image_label)

        # Layout chính HBox
        main_layout.addLayout(image_layout, 70)
        main_layout.addLayout(control_layout, 30)

        # ========= Layout tổng có HEADER =========
        final_layout = QVBoxLayout()
        final_layout.addWidget(self.header_label)
        final_layout.addLayout(main_layout)

        self.setLayout(final_layout)
        self.current_img_path = None
        
        # Camera variables
        self.camera = None
        self.timer = None
        self.camera_active = False

    def open_img(self):
        # Stop camera if running
        if self.camera_active:
            self.stop_camera()
        
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh", "", "Image Files (*.jpg *.jpeg *.png)"
        )
        if path:
            self.current_img_path = path

            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w, ch = img.shape
            bytes_per_line = ch * w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)

            pix = QPixmap.fromImage(qimg).scaled(
                500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

            self.image_label.setPixmap(pix)
            self.result_label.setText("Result: Ready")

    def predict(self):
        if not self.current_img_path:
            self.result_label.setText("⚠ Please select an image first!")
            return

        try:
            # Hiển thị trạng thái đang xử lý
            self.result_label.setText("Processing...")
            QApplication.processEvents()  # Cập nhật UI
            
            img, label, conf = predict_image(self.current_img_path)
            self.result_label.setText(f"{label} ({conf*100:.1f}%)")
            
        except Exception as e:
            # Hiển thị lỗi ngắn gọn trên giao diện
            error_str = str(e)
            if "Could not deserialize" in error_str:
                short_msg = "Lỗi phiên bản Model (Keras version mismatch)"
            elif "No such file" in error_str:
                short_msg = "Không tìm thấy file model hoặc ảnh"
            else:
                # Lấy 50 ký tự đầu của lỗi để hiển thị
                short_msg = f"Lỗi: {error_str[:50]}..."
            
            self.result_label.setText(short_msg)
            
            # In lỗi chi tiết ra console để debug
            print("-" * 30)
            print("CHI TIẾT LỖI:")
            print(error_str)
            print("-" * 30)
            import traceback
            traceback.print_exc()

    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        """Start camera capture"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.result_label.setText("⚠ Cannot open camera!")
                return
            
            self.camera_active = True
            self.btn_camera.setText("Dừng Camera")
            self.btn_camera.setStyleSheet("background-color: #f57c00;")
            self.result_label.setText("Camera: Active")
            
            # Create timer for updating frames
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)  # Update every 30ms (~33 FPS)
            
        except Exception as e:
            self.result_label.setText(f"⚠ Camera error: {str(e)[:50]}")
            print(f"Camera error: {e}")

    def stop_camera(self):
        """Stop camera capture"""
        self.camera_active = False
        if self.timer:
            self.timer.stop()
        if self.camera:
            self.camera.release()
        
        # Set black screen instead of frozen frame
        black_pixmap = QPixmap(500, 500)
        black_pixmap.fill(Qt.black)
        self.image_label.setPixmap(black_pixmap)
        
        self.btn_camera.setText("Mở Camera")
        self.btn_camera.setStyleSheet("")
        self.result_label.setText("Camera: Stopped")

    def update_frame(self):
        """Update frame from camera"""
        if not self.camera_active or not self.camera:
            return
        
        ret, frame = self.camera.read()
        if not ret:
            self.result_label.setText("⚠ Failed to read from camera")
            return
        
        # Predict on frame
        try:
            label, conf = predict_frame(frame)
            
            # Convert frame to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pix)
            
            # Update result
            self.result_label.setText(f"Camera: {label} ({conf*100:.1f}%)")
            
        except Exception as e:
            print(f"Frame prediction error: {e}")

    def exit_app(self):
        """Exit the application"""
        if self.camera_active:
            self.stop_camera()
        QApplication.quit()

    def closeEvent(self, event):
        """Handle window close event"""
        if self.camera_active:
            self.stop_camera()
        event.accept()


# ===== Run App =====
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    with open("style.qss", "r", encoding="utf-8") as f:
        app.setStyleSheet(f.read())
    
    window = App()
    window.show()
    sys.exit(app.exec_())
