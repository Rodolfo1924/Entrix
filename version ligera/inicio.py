import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt
from registro_rostros import RegistroAutomaticoUI
from reconocimiento import ProFaceAuth

class InicioUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bienvenido al sistema de reconocimiento facial")
        self.setFixedSize(1200, 750)
        self.setStyleSheet("""
            QWidget { background: #7b2ff2; }
            QLabel { color: #fff0f6; font-size: 28px; font-weight: bold; }
            QPushButton { background-color: #f357a8; color: white; font-size: 22px; border-radius: 10px; border: 2px solid #a4508b; padding: 16px; }
            QPushButton:hover { background-color: #a4508b; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(120, 80, 120, 80)
        layout.setSpacing(60)
        self.label = QLabel("Sistema de Autenticación Facial")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumHeight(80)
        layout.addWidget(self.label)
        layout.addSpacing(40)
        self.btn_registro = QPushButton("Registrar nuevo usuario")
        self.btn_registro.setMinimumHeight(70)
        self.btn_registro.setMinimumWidth(350)
        self.btn_registro.clicked.connect(self.abrir_registro)
        layout.addWidget(self.btn_registro, alignment=Qt.AlignCenter)
        layout.addSpacing(30)
        self.btn_reconocimiento = QPushButton("Iniciar reconocimiento")
        self.btn_reconocimiento.setMinimumHeight(70)
        self.btn_reconocimiento.setMinimumWidth(350)
        self.btn_reconocimiento.clicked.connect(self.abrir_reconocimiento)
        layout.addWidget(self.btn_reconocimiento, alignment=Qt.AlignCenter)
        layout.addSpacing(40)
    def abrir_registro(self):
        self.registro_win = RegistroAutomaticoUI(self)
        self.registro_win.show()
    def abrir_reconocimiento(self):
        self.recon_win = ProFaceAuth()
        self.recon_win.show()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    inicio = InicioUI()
    inicio.show()
    sys.exit(app.exec_())
