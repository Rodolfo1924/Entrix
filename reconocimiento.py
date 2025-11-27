import os
import sys
import cv2
import numpy as np
import serial
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QMessageBox, QProgressBar, QFrame
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
from sklearn.preprocessing import LabelEncoder
import glob

ESP32_PORT = "COM3"
ESP32_BAUD = 115200
CONFIDENCE_THRESHOLD = 3500  # umbral configurable

class ProFaceAuth(QWidget):
    def __init__(self, onFinish=None):
        super().__init__()
        self.onFinish = onFinish

        # ================================
        #       SERIAL AL ESP32
        # ================================
        self.serial = None
        try:
            self.serial = serial.Serial(ESP32_PORT, ESP32_BAUD, timeout=1)
            print("✔ ESP32 conectado en", ESP32_PORT)
        except Exception as e:
            print("⚠ No se pudo abrir COM3:", e)

        self.setWindowTitle("Sistema de Autenticación Segura")
        self.setFixedSize(1000, 700)
        self.setStyleSheet("""
            QWidget { background: #7b2ff2; }
            QLabel { color: #fff0f6; }
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

        # ================================
        #          CÁMARA
        # ================================
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
        self.haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # ====================================================
    #               Cargar Modelo
    # ====================================================
    def load_model(self):
        rostros_path = 'entrix/rostros/'
        face_images, face_labels = [], []

        for usuario in os.listdir(rostros_path):
            user_path = os.path.join(rostros_path, usuario)
            for img_path in glob.glob(os.path.join(user_path, '*.jpg')):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (160, 160))
                face_images.append(img)
                face_labels.append(usuario)

        encoder = LabelEncoder()
        labels_encoded = encoder.fit_transform(face_labels) if face_labels else []

        model = cv2.face.EigenFaceRecognizer_create()
        if len(face_images) > 0:
            model.train(face_images, np.array(labels_encoded))

        return model, encoder

    # ====================================================
    #             Detección Frame a Frame
    # ====================================================
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haarcascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = gray[y:y+h, x:x+w]

            try:
                face_resized = cv2.resize(face, (160, 160))
                label_pred, confidence = self.model.predict(face_resized)

                if confidence < CONFIDENCE_THRESHOLD:
                    self.final_name = self.encoder.inverse_transform([label_pred])[0]
                else:
                    self.final_name = "desconocido"
            except Exception:
                self.final_name = "desconocido"

            if not self.progress.isVisible():
                if self.final_name != "desconocido":
                    self.status.setText(f"Rostro detectado: {self.final_name}")
                    self.status.setStyleSheet("color: #f357a8;")
                else:
                    self.status.setText("Desconocido")
                    self.status.setStyleSheet("color: #a4508b;")
                self.progress.setVisible(True)
                self.progress.setValue(0)
                self.progress_timer.start(70)
        else:
            self.status.setText("Esperando detección...")
            self.progress.setVisible(False)
            self.progress_timer.stop()

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        disp_img = cv2.resize(rgb_img, (self.video_label.width(), self.video_label.height()))
        q_img = QImage(disp_img.data, disp_img.shape[1], disp_img.shape[0], disp_img.shape[1] * 3, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    # ====================================================
    #     Cuando la barra llega a 100%
    # ====================================================
    def update_progress(self):
        value = self.progress.value() + 5
        if value <= 100:
            self.progress.setValue(value)
        else:
            self.progress_timer.stop()
            if self.final_name != "desconocido":
                if self.serial:
                    self.serial.write(b"VALID\n")
                    print("✔ Enviado OK al ESP32")
                if self.onFinish:
                    self.onFinish("VALID")
                QMessageBox.information(self, "Acceso", f"Bienvenido {self.final_name}.")
            else:
                if self.serial:
                    self.serial.write(b"INVALID\n")
                    print("❌ Enviado FAIL al ESP32")
                if self.onFinish:
                    self.onFinish("INVALID")
                QMessageBox.warning(self, "Acceso denegado", "Identidad desconocida.")

            # 🔴 Liberar COM3 inmediatamente
            if self.serial:
                try:
                    self.serial.close()
                    print("Puerto COM3 liberado")
                except:
                    pass
                self.serial = None

            self.close()

    def closeEvent(self, event):
        self.cap.release()
        self.timer.stop()
        self.progress_timer.stop()
        if self.serial:
            try:
                self.serial.close()
                print("Puerto COM3 liberado en closeEvent")
            except:
                pass
            self.serial = None
        event.accept()
