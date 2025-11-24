import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QProgressBar, QFrame
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
from sklearn.preprocessing import LabelEncoder
import glob

class ProFaceAuth(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Autenticación Segura")
        self.setFixedSize(1000, 700)
        self.setStyleSheet("""
            QWidget { background: #7b2ff2; }
            QLabel { color: #fff0f6; }
            QPushButton { background-color: #f357a8; color: white; font-size: 20px; padding: 14px; border-radius: 8px; border: 2px solid #a4508b; }
            QPushButton:hover { background-color: #a4508b; }
            QPushButton:disabled { background-color: #e0aaff; color: #fff0f6; }
            QProgressBar { border: none; border-radius: 6px; background-color: #a4508b; height: 24px; text-align: center; color: white; }
            QProgressBar::chunk { background-color: #f357a8; border-radius: 6px; }
            QFrame { background-color: #fff0f6; border-radius: 20px; border: 2px solid #a4508b; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(120, 60, 120, 60)
        layout.setSpacing(30)
        self.title = QLabel("Sistema de Autenticación Segura")
        self.title.setFont(QFont("Segoe UI", 38, QFont.Bold))
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("color: #f357a8;")
        layout.addWidget(self.title)
        card = QFrame()
        card_layout = QVBoxLayout(card)
        self.video_label = QLabel()
        self.video_label.setFixedSize(800, 500)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border-radius: 20px; background-color: #e0aaff;")
        card_layout.addWidget(self.video_label)
        layout.addWidget(card, alignment=Qt.AlignCenter)
        self.status = QLabel("Esperando detección...")
        self.status.setFont(QFont("Segoe UI", 24))
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("color: #a4508b;")
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.progress.setMinimumHeight(30)
        layout.addWidget(self.progress)
        self.confirm_btn = QPushButton("Confirmar Identidad")
        self.confirm_btn.setMinimumHeight(50)
        self.confirm_btn.setMinimumWidth(350)
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.setVisible(False)
        self.confirm_btn.clicked.connect(self.confirm_identity)
        layout.addWidget(self.confirm_btn, alignment=Qt.AlignCenter)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "No se pudo acceder a la cámara.")
            sys.exit(1)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.final_name = "desconocido"
        self.model, self.encoder = self.load_model()
        self.haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    def load_model(self):
        rostros_path = 'entrix/rostros/'
        face_images = []
        face_labels = []
        for usuario in os.listdir(rostros_path):
            user_path = os.path.join(rostros_path, usuario)
            for img_path in glob.glob(os.path.join(user_path, '*.jpg')):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (160, 160))
                face_images.append(img)
                face_labels.append(usuario)
        face_images = [np.array(img, dtype=np.uint8) for img in face_images]
        face_labels = np.array(face_labels)
        encoder = LabelEncoder()
        labels_encoded = encoder.fit_transform(face_labels)
        model = cv2.face.EigenFaceRecognizer_create()
        if len(face_images) > 0:
            model.train(face_images, labels_encoded)
        return model, encoder
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        faces_small = self.haarcascade.detectMultiScale(small_gray, 1.3, 5)
        faces = [(x * 2, y * 2, w * 2, h * 2) for (x, y, w, h) in faces_small]
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = gray[y:y+h, x:x+w]
            try:
                face_resized = cv2.resize(face, (160, 160))
                label_pred, confidence = self.model.predict(face_resized)
                if confidence < 3500:
                    self.final_name = self.encoder.inverse_transform([label_pred])[0]
                else:
                    self.final_name = "desconocido"
            except Exception:
                self.final_name = "desconocido"
            if self.final_name != "desconocido" and not self.progress.isVisible():
                self.status.setText(f"Rostro detectado: {self.final_name}")
                self.status.setStyleSheet("color: #f357a8;")
                self.progress.setVisible(True)
                self.progress.setValue(0)
                self.progress_timer.start(70)
            elif self.final_name == "desconocido":
                self.status.setText("Desconocido")
                self.status.setStyleSheet("color: #a4508b;")
                self.progress.setVisible(False)
                self.confirm_btn.setVisible(False)
                self.confirm_btn.setEnabled(False)
                self.progress_timer.stop()
        else:
            self.status.setText("Esperando detección...")
            self.status.setStyleSheet("color: #a4508b;")
            self.progress.setVisible(False)
            self.confirm_btn.setVisible(False)
            self.confirm_btn.setEnabled(False)
            self.progress_timer.stop()
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        disp_img = cv2.resize(rgb_img, (self.video_label.width(), self.video_label.height()))
        q_img = QImage(disp_img.data, disp_img.shape[1], disp_img.shape[0], ch * disp_img.shape[1], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap)
    def update_progress(self):
        value = self.progress.value() + 5
        if value <= 100:
            self.progress.setValue(value)
        else:
            self.progress_timer.stop()
            self.status.setText("Verificación lista")
            self.status.setStyleSheet("color: #f357a8;")
            self.confirm_btn.setVisible(True)
            self.confirm_btn.setEnabled(True)
    def confirm_identity(self):
        if self.final_name != "desconocido":
            QMessageBox.information(self, "Acceso", f"Bienvenido {self.final_name}.")
        else:
            QMessageBox.warning(self, "Acceso denegado", "Identidad desconocida.")
    def closeEvent(self, event):
        self.cap.release()
        self.timer.stop()
        event.accept()
