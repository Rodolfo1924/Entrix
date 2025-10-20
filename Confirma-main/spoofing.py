import sys, time, os
import cv2
import numpy as np
import serial
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QProgressBar, QFrame
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# ====== Conexión con Arduino ======
try:
    arduino = serial.Serial('COM5', 9600, timeout=1)
    time.sleep(2)
    print("✅ Conectado a Arduino")
except Exception as e:
    print(f"⚠️ No se pudo conectar al Arduino: {e}")
    arduino = None

# ====== Reconocimiento facial (se mantiene por compatibilidad) ======
facenet = FaceNet()
faces_embeddings = np.load("vectores_rostros_4personas.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))


class ProFaceAuth(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Autenticación Segura")
        self.showFullScreen()

        # ====== Estilos ======
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #0f2027, stop:0.5 #203a43, stop:1 #2c5364);
            }
            QLabel { color: #ecf0f1; }
            QPushButton {
                background-color: #34495e;
                color: white;
                font-size: 18px;
                padding: 12px;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #2c3e50; }
            QPushButton:disabled {
                background-color: #7f8c8d;
                color: #bdc3c7;
            }
        """)

        # ====== Layout principal ======
        layout = QVBoxLayout(self)
        layout.setContentsMargins(100, 60, 100, 60)
        layout.setSpacing(25)

        # Título
        self.title = QLabel("Sistema de Autenticación Segura")
        self.title.setFont(QFont("Segoe UI", 34, QFont.Bold))
        self.title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title)

        # Marco cámara
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #111;
                border-radius: 20px;
                border: 2px solid #2c3e50;
            }
        """)
        card_layout = QVBoxLayout(card)

        self.video_label = QLabel()
        self.video_label.setFixedSize(700, 450)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border-radius: 20px; background-color: black;")
        card_layout.addWidget(self.video_label)

        layout.addWidget(card, alignment=Qt.AlignCenter)

        # Estado
        self.status = QLabel("Esperando detección...")
        self.status.setFont(QFont("Segoe UI", 20))
        self.status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status)

        # Barra de progreso estilo moderno
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 6px;
                background-color: #2c3e50;
                height: 18px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #1abc9c;
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.progress)

        # Botón Confirmar (ya no se usará en fraude, pero se deja para estructura)
        self.confirm_btn = QPushButton("Confirmar Identidad")
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.setVisible(False)
        self.confirm_btn.clicked.connect(self.confirm_identity)
        layout.addWidget(self.confirm_btn, alignment=Qt.AlignCenter)

        # ====== Cámara ======
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "No se pudo acceder a la cámara.")
            sys.exit(1)

        # Timer cámara
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)

        # Timer progreso
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)

        # Estado reconocimiento
        self.final_name = "desconocido"

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haarcascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0 and not self.progress.isVisible():
            # Inicia la barra de progreso cuando se detecta un rostro
            self.status.setText("Rostro detectado, verificando...")
            self.status.setStyleSheet("color: #1abc9c;")
            self.progress.setVisible(True)
            self.progress.setValue(0)
            self.progress_timer.start(70)

        elif len(faces) == 0:
            self.status.setText("Esperando detección...")
            self.status.setStyleSheet("color: #ecf0f1;")
            self.progress.setVisible(False)
            self.confirm_btn.setVisible(False)
            self.confirm_btn.setEnabled(False)
            self.progress_timer.stop()

        # Mostrar video
        h, w, ch = rgb_img.shape
        q_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def update_progress(self):
        value = self.progress.value() + 5
        if value <= 100:
            self.progress.setValue(value)
        else:
            self.progress_timer.stop()
            # 🚨 Aquí en lugar de éxito, mostramos fraude
            self.status.setText("⚠️ Advertencia: Intento de fraude detectado")
            self.status.setStyleSheet("color: orange; font-weight: bold;")
            QMessageBox.warning(self, "Fraude", "❌ Se detectó un intento de fraude. Acceso bloqueado.")
            self.progress.setVisible(False)

    def confirm_identity(self):
        # No se usa en este modo, pero lo dejamos definido
        QMessageBox.warning(self, "Acceso denegado", "❌ Intento de fraude detectado.")

    def close_door(self):
        if arduino is not None:
            arduino.write("close\r\n".encode())
            print(">>> Enviado: close")

    def closeEvent(self, event):
        self.cap.release()
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProFaceAuth()
    window.show()
    sys.exit(app.exec_())
