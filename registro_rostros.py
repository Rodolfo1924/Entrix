import os
import cv2
import time
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QProgressBar,
    QLineEdit, QMessageBox, QFrame, QApplication, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from theme import STYLE_GLOBAL


class RegistroAutomaticoUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(STYLE_GLOBAL)

        self.parent_window = parent

        # =============================
        #   CONFIG GENERAL
        # =============================
        self.setWindowTitle("Registro de Nuevo Usuario")
        self.setFixedSize(1100, 750)
        self.setStyleSheet("""
            QWidget { 
                background-color: #07101f;
                font-family: Segoe UI;
                color: white;
            }
        """)

        # =============================
        #   PANEL PRINCIPAL
        # =============================
        panel = QFrame(self)
        panel.setGeometry(60, 40, 980, 660)
        panel.setStyleSheet("""
            QFrame {
                background-color: #0f1622;
                border-radius: 18px;
                border: 2px solid #c0c7d1;
            }
        """)

        wrapper = QVBoxLayout(panel)
        wrapper.setContentsMargins(30, 30, 30, 30)
        wrapper.setSpacing(25)

        # ==========================================================
        #   TOP BAR (Regresar + Título)
        # ==========================================================
        top_bar = QHBoxLayout()
        top_bar.setSpacing(15)

        btn_back = QPushButton("← Regresar")
        btn_back.setFixedWidth(140)
        btn_back.setStyleSheet("""
            QPushButton {
                background-color: #2a4f9e;
                border: 1px solid #c0c7d1;
                border-radius: 8px;
                padding: 8px;
                font-size: 16px;
                color: white;
            }
            QPushButton:hover {
                background-color: #1b356b;
            }
        """)
        btn_back.clicked.connect(self.regresar)
        top_bar.addWidget(btn_back, alignment=Qt.AlignLeft)

        title = QLabel("Registro de Usuario")
        title.setStyleSheet("font-size: 30px; font-weight: bold;")
        top_bar.addWidget(title, alignment=Qt.AlignCenter)

        top_bar.addStretch()
        wrapper.addLayout(top_bar)

        # ==========================================================
        #   CONTENIDO PRINCIPAL (2 columnas)
        # ==========================================================
        content = QHBoxLayout()
        content.setSpacing(40)
        wrapper.addLayout(content)

        # ----------------------------------------------------------
        #   COLUMNA IZQUIERDA — FORMULARIO
        # ----------------------------------------------------------
        left_col = QVBoxLayout()
        left_col.setSpacing(18)

        subtitle = QLabel("Captura automática del rostro")
        subtitle.setStyleSheet("font-size: 17px; color: #c0c7d1;")
        left_col.addWidget(subtitle)

        # Nombre
        lbl_nombre = QLabel("Nombre del usuario:")
        lbl_nombre.setStyleSheet("font-size: 18px;")
        left_col.addWidget(lbl_nombre)

        self.input_nombre = QLineEdit()
        self.input_nombre.setMinimumHeight(38)
        self.input_nombre.setStyleSheet("""
            QLineEdit {
                background-color: #f2f4f7;
                color: #000;
                border-radius: 6px;
                padding: 8px;
                border: 1px solid #c0c7d1;
                font-size: 18px;
            }
        """)
        left_col.addWidget(self.input_nombre)

        # Botón captura
        self.btn_capturar = QPushButton("Iniciar Captura Automática")
        self.btn_capturar.setMinimumHeight(45)
        self.btn_capturar.setStyleSheet("""
            QPushButton {
                background-color: #2a4f9e;
                border-radius: 10px;
                border: 1px solid #c0c7d1;
                color: white;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #1b356b;
            }
        """)
        self.btn_capturar.clicked.connect(self.capturar_rostros_auto)
        left_col.addWidget(self.btn_capturar)

        # Progreso
        self.progress = QProgressBar()
        self.progress.setMaximum(20)
        self.progress.setValue(0)
        self.progress.setStyleSheet("""
            QProgressBar {
                background-color: #1a202c;
                border-radius: 10px;
                height: 26px;
                border: 1px solid #c0c7d1;
                color: white;
                font-size: 14px;
            }
            QProgressBar::chunk {
                background-color: #2a4f9e;
                border-radius: 10px;
            }
        """)
        left_col.addWidget(self.progress)

        # Estado
        self.status = QLabel("")
        self.status.setStyleSheet("font-size: 17px; color: #c0c7d1; margin-top: 5px;")
        left_col.addWidget(self.status)

        left_col.addStretch()
        content.addLayout(left_col)

        # ----------------------------------------------------------
        #   COLUMNA DERECHA — VIDEO
        # ----------------------------------------------------------
        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        self.video_label = QLabel()
        self.video_label.setFixedSize(580, 430)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #0c0f14;
                border: 2px solid #c0c7d1;
                border-radius: 12px;
            }
        """)
        right_col.addWidget(self.video_label, alignment=Qt.AlignCenter)

        right_col.addStretch()
        content.addLayout(right_col)

        # ===========================
        # CÁMARA
        # ===========================
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)

        self.n_imagenes = 20
        self.capturadas = 0
        self.capturando = False

        self.haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # =====================================================
    #   ACTUALIZAR VIDEO
    # =====================================================
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.video_label.width(), self.video_label.height()))
        qimage = QImage(resized.data, resized.shape[1], resized.shape[0], resized.strides[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimage))

        if self.capturando and self.capturadas < self.n_imagenes:
            self.detectar_y_guardar(frame)

        if self.capturadas >= self.n_imagenes and self.capturando:
            self.capturando = False
            self.status.setText("✔ Captura completada")
            QMessageBox.information(self, "Registro", "Rostros capturados correctamente.")
            from model_loader import FaceModel
            FaceModel._model = None   # limpiar cache
            FaceModel._encoder = None

            self.regresar()

    # =====================================================
    #   GUARDAR ROSTROS
    # =====================================================
    def detectar_y_guardar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haarcascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            rostro = cv2.resize(gray[y:y+h, x:x+w], (160, 160))

            nombre = self.input_nombre.text().strip()
            path = os.path.join("entrix", "rostros", nombre)
            os.makedirs(path, exist_ok=True)

            cv2.imwrite(os.path.join(path, f"rostro_{self.capturadas+1}.jpg"), rostro)

            self.capturadas += 1
            self.progress.setValue(self.capturadas)
            self.status.setText(f"Capturando {self.capturadas}/{self.n_imagenes}")
            QApplication.processEvents()

            time.sleep(0.18)

    # =====================================================
    #   INICIAR CAPTURA
    # =====================================================
    def capturar_rostros_auto(self):
        nombre = self.input_nombre.text().strip()
        if not nombre:
            QMessageBox.warning(self, "Registro", "Ingrese un nombre válido.")
            return

        self.capturadas = 0
        self.progress.setValue(0)

        self.capturando = True
        self.status.setText("Iniciando...")

    # =====================================================
    #   REGRESAR
    # =====================================================
    def regresar(self):
        self.close()
        if self.parent_window:
            self.parent_window.show()

    # =====================================================
    #   CERRAR
    # =====================================================
    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        self.timer.stop()
        event.accept()
