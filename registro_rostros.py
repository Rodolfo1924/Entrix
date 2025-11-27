import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QProgressBar, QLineEdit, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
import time
from PyQt5.QtWidgets import QApplication

class RegistroAutomaticoUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Registro de nuevo usuario")
        self.setFixedSize(800, 600)
        self.setStyleSheet("""
            QWidget { background: #fff0f6; }
            QLabel { color: #a4508b; font-size: 24px; }
            QPushButton { background-color: #f357a8; color: white; font-size: 20px; border-radius: 8px; border: 2px solid #a4508b; padding: 12px; }
            QPushButton:hover { background-color: #a4508b; }
            QLineEdit { font-size: 20px; border-radius: 6px; border: 2px solid #a4508b; padding: 8px; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(80, 60, 80, 60)
        layout.setSpacing(30)
        self.label = QLabel("Ingrese el nombre del nuevo usuario:")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        self.input_nombre = QLineEdit()
        self.input_nombre.setMinimumHeight(40)
        layout.addWidget(self.input_nombre, alignment=Qt.AlignCenter)
        self.video_label = QLabel()
        self.video_label.setFixedSize(500, 350)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border-radius: 12px; background-color: #e0aaff;")
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setMaximum(20)
        self.progress.setVisible(True)
        self.progress.setMinimumHeight(30)
        layout.addWidget(self.progress)
        self.btn_capturar = QPushButton("Iniciar captura automática")
        self.btn_capturar.setMinimumHeight(50)
        self.btn_capturar.setMinimumWidth(300)
        self.btn_capturar.clicked.connect(self.capturar_rostros_auto)
        layout.addWidget(self.btn_capturar, alignment=Qt.AlignCenter)
        self.status = QLabel("")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setMinimumHeight(40)
        layout.addWidget(self.status)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)
        self.capturadas = 0
        self.n_imagenes = 20
        self.haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.capturando = False
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        disp_img = cv2.resize(rgb_img, (self.video_label.width(), self.video_label.height()))
        q_img = QImage(disp_img.data, disp_img.shape[1], disp_img.shape[0], ch * disp_img.shape[1], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap)
        if self.capturando and self.capturadas < self.n_imagenes:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            faces_small = self.haarcascade.detectMultiScale(small_gray, 1.3, 5)
            faces = [(x * 2, y * 2, w * 2, h * 2) for (x, y, w, h) in faces_small]
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (160, 160))
                nombre = self.input_nombre.text().strip()
                if nombre:
                    user_dir = os.path.join(os.getcwd(), 'entrix', 'rostros', nombre)
                    os.makedirs(user_dir, exist_ok=True)
                    img_path = os.path.join(user_dir, f"rostro_{self.capturadas+1}.jpg")
                    cv2.imwrite(img_path, face_resized)
                    self.capturadas += 1
                    self.progress.setValue(self.capturadas)
                    self.status.setText(f"Capturando rostro... ({self.capturadas}/{self.n_imagenes})")
                    QApplication.processEvents()
                    time.sleep(0.2)
        if self.capturadas >= self.n_imagenes and self.capturando:
            self.capturando = False
            self.status.setText("¡Captura finalizada!")
            QMessageBox.information(self, "Registro", "Rostros capturados correctamente.")
            self.close()
    def capturar_rostros_auto(self):
        nombre = self.input_nombre.text().strip()
        if not nombre:
            QMessageBox.warning(self, "Registro", "Ingrese el nombre del usuario.")
            return
        self.capturando = True
        self.status.setText("Iniciando captura automática...")
    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        self.timer.stop()
        event.accept()
