import sys, time, os
import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QProgressBar, QFrame, QInputDialog
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
from registro_usuario import registrar_nuevo_usuario, RegistroUsuarioUI

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# ====== Reconocimiento facial ======
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

        # ====== Estilos (colores sólidos, rosas y violetas) ======
        self.setStyleSheet("""
            QWidget {
                background: #7b2ff2;
            }
            QLabel {
                color: #fff0f6;
            }
            QPushButton {
                background-color: #f357a8;
                color: white;
                font-size: 18px;
                padding: 12px;
                border-radius: 8px;
                border: 2px solid #a4508b;
            }
            QPushButton:hover {
                background-color: #a4508b;
            }
            QPushButton:disabled {
                background-color: #e0aaff;
                color: #fff0f6;
            }
            QProgressBar {
                border: none;
                border-radius: 6px;
                background-color: #a4508b;
                height: 18px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #f357a8;
                border-radius: 6px;
            }
            QFrame {
                background-color: #fff0f6;
                border-radius: 20px;
                border: 2px solid #a4508b;
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
        self.title.setStyleSheet("color: #f357a8;")
        layout.addWidget(self.title)

        # Marco cámara
        card = QFrame()
        card_layout = QVBoxLayout(card)

        self.video_label = QLabel()
        self.video_label.setFixedSize(700, 450)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border-radius: 20px; background-color: #e0aaff;")
        card_layout.addWidget(self.video_label)

        layout.addWidget(card, alignment=Qt.AlignCenter)

        # Estado
        self.status = QLabel("Esperando detección...")
        self.status.setFont(QFont("Segoe UI", 20))
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("color: #a4508b;")
        layout.addWidget(self.status)

        # Barra de progreso
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Botón Confirmar
        self.confirm_btn = QPushButton("Confirmar Identidad")
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.setVisible(False)
        self.confirm_btn.clicked.connect(self.confirm_identity)
        layout.addWidget(self.confirm_btn, alignment=Qt.AlignCenter)

        # Botón Registrar usuario (solo para administrador)
        self.register_btn = QPushButton("Registrar usuario")
        self.register_btn.setStyleSheet("background-color: #a4508b; color: white; font-size: 18px; border-radius: 8px; border: 2px solid #f357a8;")
        self.register_btn.clicked.connect(self.registro_desde_gui)
        layout.addWidget(self.register_btn, alignment=Qt.AlignCenter)

        # ====== Cámara ======
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "No se pudo acceder a la cámara.")
            sys.exit(1)

        # Timer cámara
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)  # Procesa cada 100ms (10 fps)

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
        # Detectar en imagen reducida
        small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        faces_small = haarcascade.detectMultiScale(small_gray, 1.3, 5)
        faces = [(x * 2, y * 2, w * 2, h * 2) for (x, y, w, h) in faces_small]

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = rgb_img[y:y+h, x:x+w]

            try:
                face_resized = cv2.resize(face, (160, 160))
                face_resized = np.expand_dims(face_resized, axis=0)
                ypred = facenet.embeddings(face_resized)
                pred = model.predict(ypred)
                self.final_name = encoder.inverse_transform(pred)[0]
            except:
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

        # Mostrar video
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

    def registro_desde_gui(self):
        self.registro_win = RegistroUsuarioUI(self)
        self.registro_win.show()

    def closeEvent(self, event):
        self.cap.release()
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProFaceAuth()
    window.show()
    sys.exit(app.exec_())
