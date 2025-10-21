import cv2
import numpy as np
import pickle
import time
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QMessageBox, QProgressBar, QFrame, QApplication
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer

class RegistroUsuarioUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Registro de nuevo usuario")
        self.setFixedSize(600, 600)
        self.setStyleSheet("""
            QWidget { background: #fff0f6; }
            QLabel { color: #a4508b; font-size: 20px; }
            QPushButton { background-color: #f357a8; color: white; font-size: 18px; border-radius: 8px; border: 2px solid #a4508b; }
            QPushButton:hover { background-color: #a4508b; }
            QLineEdit { font-size: 18px; border-radius: 6px; border: 2px solid #a4508b; padding: 6px; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)

        self.label = QLabel("Ingrese el nombre del nuevo usuario:")
        layout.addWidget(self.label)
        self.input_nombre = QLineEdit()
        layout.addWidget(self.input_nombre)

        self.video_label = QLabel()
        self.video_label.setFixedSize(400, 300)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border-radius: 12px; background-color: #e0aaff;")
        layout.addWidget(self.video_label)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setMaximum(20)
        self.progress.setVisible(True)
        layout.addWidget(self.progress)

        self.btn_capturar = QPushButton("Capturar rostro")
        self.btn_capturar.clicked.connect(self.capturar_rostro)
        layout.addWidget(self.btn_capturar)

        self.status = QLabel("")
        self.status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)

        self.embeddings = []
        self.capturadas = 0
        self.n_imagenes = 20
        self.facenet = FaceNet()
        self.haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

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

    def capturar_rostro(self):
        nombre = self.input_nombre.text().strip()
        if not nombre:
            QMessageBox.warning(self, "Registro", "Ingrese el nombre del usuario.")
            return
        while self.capturadas < self.n_imagenes:
            ret, frame = self.cap.read()
            if not ret:
                continue
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            faces_small = self.haarcascade.detectMultiScale(small_gray, 1.3, 5)
            faces = [(x * 2, y * 2, w * 2, h * 2) for (x, y, w, h) in faces_small]
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = rgb_img[y:y+h, x:x+w]
                try:
                    face_resized = cv2.resize(face, (160, 160))
                    face_resized = np.expand_dims(face_resized, axis=0)
                    emb = self.facenet.embeddings(face_resized)[0]
                    self.embeddings.append(emb)
                    self.capturadas += 1
                    self.progress.setValue(self.capturadas)
                    self.status.setText(f"Capturando rostro... ({self.capturadas}/{self.n_imagenes})")
                    QApplication.processEvents()
                    time.sleep(0.2)
                except:
                    continue
        self.timer.stop()
        self.cap.release()
        self.status.setText("Procesando y guardando datos...")
        QApplication.processEvents()
        self.guardar_usuario(nombre)

    def guardar_usuario(self, nombre):
        archivo_vectores = "vectores_rostros_4personas.npz"
        archivo_modelo = "svm_model_160x160.pkl"
        # Cargar vectores existentes
        data = np.load(archivo_vectores)
        X = data['arr_0']
        Y = data['arr_1']
        # Agregar nuevos datos
        X_new = np.concatenate([X, np.array(self.embeddings)], axis=0)
        Y_new = np.concatenate([Y, np.array([nombre]*self.n_imagenes)], axis=0)
        # Guardar
        np.savez(archivo_vectores, X_new, Y_new)
        # Reentrenar SVM
        encoder = LabelEncoder()
        encoder.fit(Y_new)
        model_new = SVC(kernel='linear', probability=True)
        model_new.fit(X_new, encoder.transform(Y_new))
        with open(archivo_modelo, "wb") as f:
            pickle.dump(model_new, f)
        self.status.setText(f"Usuario '{nombre}' registrado y modelo actualizado.")
        QMessageBox.information(self, "Registro", f"Usuario '{nombre}' registrado exitosamente.")
        self.close()

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        self.timer.stop()
        event.accept()

def registrar_nuevo_usuario(nombre, n_imagenes=20,
                            archivo_vectores="vectores_rostros_4personas.npz",
                            archivo_modelo="svm_model_160x160.pkl",
                            haar_path="haarcascade_frontalface_default.xml",
                            cam_index=0):
    """
    Registra un nuevo usuario capturando imágenes desde la cámara, calculando embeddings y actualizando el modelo SVM.
    Args:
        nombre (str): Nombre del usuario a registrar.
        n_imagenes (int): Número de imágenes a capturar.
        archivo_vectores (str): Ruta al archivo de vectores .npz.
        archivo_modelo (str): Ruta al archivo del modelo SVM .pkl.
        haar_path (str): Ruta al archivo haarcascade.
        cam_index (int): Índice de la cámara.
    """
    facenet = FaceNet()
    haarcascade = cv2.CascadeClassifier(haar_path)
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    embeddings = []
    capturadas = 0
    print(f"[INFO] Coloque el rostro frente a la cámara. Se capturarán {n_imagenes} imágenes.")
    while capturadas < n_imagenes:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        faces_small = haarcascade.detectMultiScale(small_gray, 1.3, 5)
        faces = [(x * 2, y * 2, w * 2, h * 2) for (x, y, w, h) in faces_small]
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = rgb_img[y:y+h, x:x+w]
            try:
                face_resized = cv2.resize(face, (160, 160))
                face_resized = np.expand_dims(face_resized, axis=0)
                emb = facenet.embeddings(face_resized)[0]
                embeddings.append(emb)
                capturadas += 1
                print(f"Capturando rostro... ({capturadas}/{n_imagenes})")
                time.sleep(0.2)
            except:
                continue
    cap.release()
    # Cargar vectores existentes
    data = np.load(archivo_vectores)
    X = data['arr_0']
    Y = data['arr_1']
    # Agregar nuevos datos
    X_new = np.concatenate([X, np.array(embeddings)], axis=0)
    Y_new = np.concatenate([Y, np.array([nombre]*n_imagenes)], axis=0)
    # Guardar
    np.savez(archivo_vectores, X_new, Y_new)
    # Reentrenar SVM
    encoder = LabelEncoder()
    encoder.fit(Y_new)
    model_new = SVC(kernel='linear', probability=True)
    model_new.fit(X_new, encoder.transform(Y_new))
    with open(archivo_modelo, "wb") as f:
        pickle.dump(model_new, f)
    print(f"[INFO] Usuario '{nombre}' registrado y modelo actualizado.")
    return True

# Ejemplo de uso:
# registrar_nuevo_usuario("NuevoUsuario")
