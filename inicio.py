import sys
import serial
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QFrame, QHBoxLayout, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

from registro_rostros import RegistroAutomaticoUI
from reconocimiento import ProFaceAuth
from theme import STYLE_GLOBAL


# =========================================
#   HILO SERIAL
# =========================================
class SerialListener(QThread):
    start_signal = pyqtSignal()
    status_signal = pyqtSignal(str)

    def __init__(self, port="COM3", baud=115200):
        super().__init__()
        self.port = port
        self.baud = baud
        self.serial = None
        self.running = True

    def run(self):
        try:
            self.serial = serial.Serial(self.port, self.baud, timeout=1)
            self.status_signal.emit(f"Conectado a {self.port}")
        except Exception as e:
            self.status_signal.emit(f"ERROR al abrir {self.port}: {e}")
            return

        while self.running:
            try:
                if self.serial.in_waiting:
                    line = self.serial.readline().decode(errors="ignore").strip()
                    self.status_signal.emit(f"Serial <- {line}")

                    if line.upper() == "START":
                        self.start_signal.emit()

            except Exception as e:
                self.status_signal.emit(f"ERROR serial: {e}")
                break

    def stop(self):
        self.running = False
        try:
            if self.serial and self.serial.is_open:
                self.serial.close()
        except:
            pass
        self.quit()
        self.wait()


# =========================================
#   INTERFAZ PRINCIPAL (DASHBOARD)
# =========================================
class InicioUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(STYLE_GLOBAL)

        self.setWindowTitle("Sistema de Autenticación Facial")
        self.setFixedSize(1100, 650)

        # ====================================
        # ESTILO
        # ====================================
        
        # ============================
        #   LAYOUT GENERAL
        # ============================
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(25)

        # ============================================================
        #   COLUMNA IZQUIERDA — ACCIONES
        # ============================================================
        left_panel = QFrame()
        left_panel.setObjectName("panel")
        left_panel.setStyleSheet("QFrame#panel { background-color:#0f1622; border-radius:18px; border:2px solid #c0c7d1; }")
        left_panel.setFixedWidth(330)

        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(25, 25, 25, 25)
        left_layout.setSpacing(30)

        title = QLabel("Menú Principal")
        title.setProperty("class", "title")
        left_layout.addWidget(title)

        # --- BOTONES ---
        btn_reg = QPushButton("Registrar Nuevo Usuario")
        btn_reg.clicked.connect(self.abrir_registro)
        left_layout.addWidget(btn_reg)

        btn_recon = QPushButton("Iniciar Reconocimiento")
        btn_recon.clicked.connect(self.abrir_reconocimiento)
        left_layout.addWidget(btn_recon)

        # --- ESTADO SERIAL ---
        self.debug_label = QLabel("Esperando conexión...")
        self.debug_label.setProperty("class", "subtitle")
        left_layout.addWidget(self.debug_label)

        left_layout.addStretch()

        main_layout.addWidget(left_panel)

        # ============================================================
        #   COLUMNA DERECHA — DASHBOARD
        # ============================================================
        dashboard = QFrame()
        dashboard.setObjectName("panel")
        dashboard.setStyleSheet("QFrame#panel { background-color:#0f1622; border-radius:18px; border:2px solid #c0c7d1; }")
        
        dash_layout = QVBoxLayout(dashboard)
        dash_layout.setContentsMargins(30, 30, 30, 30)
        dash_layout.setSpacing(25)

        dash_title = QLabel("Dashboard del Sistema")
        dash_title.setProperty("class", "title")
        dash_layout.addWidget(dash_title)

        # ------------------------
        # TARJETAS
        # ------------------------
        grid = QGridLayout()
        grid.setSpacing(20)

        # Usuarios registrados
        self.card_users = self.crear_card("Usuarios Registrados", self.contar_usuarios())
        grid.addWidget(self.card_users, 0, 0)

        # Último acceso
        self.card_last = self.crear_card("Último usuario reconocido", "—")
        grid.addWidget(self.card_last, 0, 1)

        # Estado de cámara
        cam_state = "Lista" if self.camara_detectada() else "No detectada"
        self.card_cam = self.crear_card("Estado de cámara", cam_state)
        grid.addWidget(self.card_cam, 1, 0)

        # Estado del modelo
        model_state = "Cargado" if os.path.exists("entrix/modelo/embeddings.npz") else "No entrenado"
        self.card_model = self.crear_card("Estado del modelo", model_state)
        grid.addWidget(self.card_model, 1, 1)

        dash_layout.addLayout(grid)

        main_layout.addWidget(dashboard)

        # ============================
        #   SERIAL
        # ============================
        self.serial_thread = None
        self.iniciar_serial_listener()

    # ============================================================
    #   TARJETAS
    # ============================================================
    def crear_card(self, titulo, valor):
        card = QFrame()
        card.setObjectName("card")
        card.setFixedSize(300, 120)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 15, 20, 15)

        label_t = QLabel(titulo)
        label_t.setProperty("class", "card-title")

        label_v = QLabel(str(valor))
        label_v.setProperty("class", "card-value")

        layout.addWidget(label_t)
        layout.addWidget(label_v)
        layout.addStretch()

        return card

    # Lógica de conteo de usuarios
    def contar_usuarios(self):
        base = "entrix/rostros"
        if not os.path.exists(base):
            return 0
        return len(os.listdir(base))

    # Detección simple de cámara
    def camara_detectada(self):
        import cv2
        cap = cv2.VideoCapture(0)
        ok = cap.isOpened()
        cap.release()
        return ok

    # ============================================================
    #   INTERACCIONES
    # ============================================================
    def update_debug(self, msg):
        self.debug_label.setText(msg)

    def handle_start(self):
        self.abrir_reconocimiento()

    def abrir_registro(self):
        self.reg_win = RegistroAutomaticoUI(self)
        self.reg_win.show()

    def abrir_reconocimiento(self):
        if self.serial_thread:
            self.serial_thread.stop()

        self.recon_win = ProFaceAuth(onFinish=self.recon_done)
        self.recon_win.show()
        self.hide()

    def recon_done(self, resultado):
        # actualizar dashboard con el último usuario
        self.card_last.findChildren(QLabel)[1].setText(resultado)

        QTimer.singleShot(500, self.iniciar_serial_listener)
        self.show()

    # ============================================================
    #   SERIAL
    # ============================================================
    def iniciar_serial_listener(self):
        try:
            self.serial_thread = SerialListener(port="COM3", baud=115200)
            self.serial_thread.start_signal.connect(self.handle_start)
            self.serial_thread.status_signal.connect(self.update_debug)
            self.serial_thread.start()
        except Exception:
            QTimer.singleShot(8000, self.iniciar_serial_listener)

    def closeEvent(self, event):
        if self.serial_thread:
            self.serial_thread.stop()
        event.accept()


# =========================================
#   RUN
# =========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InicioUI()
    window.show()
    
    sys.exit(app.exec_())
