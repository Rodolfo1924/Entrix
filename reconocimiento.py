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
from serial_manager import SerialManager
from model_loader import FaceModel
from theme import STYLE_GLOBAL


ESP32_PORT = "COM3"
ESP32_BAUD = 115200
CONFIDENCE_THRESHOLD = 3500  # umbral configurable


class ProFaceAuth(QWidget):
    def __init__(self, onFinish=None):
        super().__init__()
        self.setStyleSheet(STYLE_GLOBAL)

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

        try:
            self.serial = SerialManager()
        except:
            self.serial = None
            print("⚠ No se pudo abrir el COM (probablemente ocupado, pero no es crítico).") ##########################################3
        
        self.model, self.encoder = FaceModel.load() #################################
        
        # ================================
        #     CONFIGURACIÓN DE VENTANA
        # ================================
        self.setWindowTitle("Sistema de Autenticación Facial")
        self.setFixedSize(1280, 800)
        

        layout = QVBoxLayout(self)
        layout.setContentsMargins(80, 40, 80, 40)
        layout.setSpacing(25)

        # ================================
        #            TÍTULO
        # ================================
        self.title = QLabel("Reconocimiento Facial")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setProperty("class", "title")
        layout.addWidget(self.title)

        # ================================
        #            TARJETA
        # ================================
        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet("""
            QFrame#card {
                background-color: #0f1622;
                border: 2px solid #c0c7d1;
                border-radius: 18px;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(30, 30, 30, 30)
        card_layout.setSpacing(25)

        # ================================
        #        VIDEO EN VIVO
        # ================================
        self.video_label = QLabel()
        self.video_label.setFixedSize(1000, 560)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: #0c0f14;
            border: 2px solid #c0c7d1;
            border-radius: 12px;
        """)

        card_layout.addWidget(self.video_label)
        layout.addWidget(card, alignment=Qt.AlignCenter)

        # ================================
        #             STATUS
        # ================================
        self.status = QLabel("Esperando detección de rostro…")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setProperty("class", "status")
        layout.addWidget(self.status)

        # ================================
        #        BARRA DE PROGRESO
        # ================================
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # ================================
        #             CÁMARA
        # ================================
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "No se pudo acceder a la cámara.")
            sys.exit(1)

        # Timers
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)

        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)

        self.final_name = "desconocido"
        self.model, self.encoder = self.load_model()
        self.haarcascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # ====================================================
    #              Cargar Modelo
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
    #            Detección Frame a Frame
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
                self.status.setText(
                    f"Detectado: {self.final_name}" if self.final_name != "desconocido"
                    else "Rostro desconocido"
                )
                self.progress.setVisible(True)
                self.progress.setValue(0)
                self.progress_timer.start(60)  # animación más suave

        else:
            self.status.setText("Esperando detección de rostro…")
            self.progress.setVisible(False)
            self.progress_timer.stop()

        # Muestra video
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        disp_img = cv2.resize(rgb_img, (self.video_label.width(), self.video_label.height()))
        q_img = QImage(
            disp_img.data, disp_img.shape[1], disp_img.shape[0],
            disp_img.shape[1] * 3, QImage.Format_RGB888
        )
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    # ====================================================
    #     Cuando la barra llega a 100%
    # ====================================================
    def update_progress(self):
        value = self.progress.value() + 4
        if value <= 100:
            self.progress.setValue(value)
        else:
            self.progress_timer.stop()
            self.finish_auth()

    # ====================================================
    #       FINALIZAR PROCESO DE ACCESO
    # ====================================================
    def finish_auth(self):
        if self.serial:
            try:
                if self.final_name != "desconocido":
                    self.serial.write(b"VALID\n")
                else:
                    self.serial.write(b"INVALID\n")
            except:
                pass

        # Resultado hacia ventana anterior
        if self.onFinish:
            self.onFinish("VALID" if self.final_name != "desconocido" else "INVALID")

        # Mensaje
        if self.final_name != "desconocido":
            # Sin cuadro de diálogo, directo al dashboard
            self.close()
            self.onFinish(self.final_name)

        else:
            QMessageBox.warning(self, "Acceso denegado", "Identidad desconocida.")

        # Liberar puerto
        if self.serial:
            try:
                self.serial.close()
            except:
                pass
            self.serial = None

        self.close()

    # ====================================================
    #           Cierre Limpio
    # ====================================================
    def closeEvent(self, event):
        self.cap.release()
        self.timer.stop()
        self.progress_timer.stop()

        if self.serial:
            try:
                self.serial.close()
            except:
                pass

        event.accept()
