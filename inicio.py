import sys
import serial
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from registro_rostros import RegistroAutomaticoUI
from reconocimiento import ProFaceAuth

# ============================
#   HILO DE ESCUCHA SERIAL
# ============================
class SerialListener(QThread):
    start_signal = pyqtSignal()        # Se emite cuando llega "START"
    status_signal = pyqtSignal(str)    # Para debug opcional

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
            self.status_signal.emit(f"ERROR: No se pudo abrir {self.port}: {e}")
            return

        while self.running:
            try:
                if self.serial.in_waiting:
                    line = self.serial.readline().decode(errors="ignore").strip()
                    self.status_signal.emit(f"Serial <- {line}")

                    if line.upper() == "START":
                        self.status_signal.emit("Comando START recibido.")
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


# ============================
#     INTERFAZ PRINCIPAL
# ============================
class InicioUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bienvenido al sistema")
        self.setFixedSize(800, 500)

        # ---- ESTILOS ----
        self.setStyleSheet("""
            QWidget { background: #7b2ff2; }
            QLabel { color: white; font-size: 28px; font-weight: bold; }
            QPushButton { background-color: #f357a8; color: white; font-size: 22px; border-radius: 10px; border: 2px solid #a4508b; padding: 16px; }
            QPushButton:hover { background-color: #a4508b; }
        """)

        layout = QVBoxLayout(self)

        self.label = QLabel("Sistema de Autenticación Facial")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.btn_registro = QPushButton("Registrar nuevo usuario")
        self.btn_registro.clicked.connect(self.abrir_registro)
        layout.addWidget(self.btn_registro)

        self.btn_reconocimiento = QPushButton("Iniciar reconocimiento (manual)")
        self.btn_reconocimiento.clicked.connect(self.abrir_reconocimiento)
        layout.addWidget(self.btn_reconocimiento)

        self.debug_label = QLabel("")
        self.debug_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.debug_label)

        # ==========================
        #  INICIAR ESCUCHA SERIAL
        # ==========================
        self.serial_thread = None
        self.iniciar_serial_listener()

    def update_debug(self, msg):
        self.debug_label.setText(msg)

    def handle_start(self):
        print(">>> Recibido START desde ESP32 — iniciando reconocimiento")
        self.abrir_reconocimiento()

    def abrir_registro(self):
        self.reg_win = RegistroAutomaticoUI(self)
        self.reg_win.show()

    def abrir_reconocimiento(self):
        # Detiene el hilo serial y cierra puerto
        if self.serial_thread:
            self.serial_thread.stop()

        # Abre la ventana de reconocimiento con callback
        self.recon_win = ProFaceAuth(onFinish=self.recon_done)
        self.recon_win.show()
        self.hide()  # oculta inicio mientras se reconoce

    def recon_done(self, resultado):
        print("Resultado:", resultado)
        # Espera medio segundo antes de reiniciar el listener
        QTimer.singleShot(500, self.iniciar_serial_listener)
        # Vuelve a mostrar la ventana principal
        self.show()

    def iniciar_serial_listener(self):
        try:
            self.serial_thread = SerialListener(port="COM3", baud=115200)
            self.serial_thread.start_signal.connect(self.handle_start)
            self.serial_thread.status_signal.connect(self.update_debug)
            self.serial_thread.start()
        except Exception as e:
            print("Error al abrir COM3:", e)
            # Reintenta después de 2 segundos
            QTimer.singleShot(8000, self.iniciar_serial_listener)


    def closeEvent(self, event):
        if self.serial_thread:
            self.serial_thread.stop()
        event.accept()


# ============================
#       EJECUCIÓN
# ============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InicioUI()
    window.show()
    sys.exit(app.exec_())
