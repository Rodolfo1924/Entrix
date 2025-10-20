import cv2 as cv
import numpy as np
import os
import serial
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

# ============ ARDUINO ============
try:
    arduino = serial.Serial('COM5', 9600, timeout=1)  # Cambia COM5 si tu Arduino está en otro puerto
    time.sleep(2)  # Esperar a que se inicie la comunicación
    print("✅ Conectado a Arduino en COM5")
except Exception as e:
    print(f"⚠️ No se pudo conectar al Arduino: {e}")
    arduino = None

# ============ RECONOCIMIENTO FACIAL ============
facenet = FaceNet()
faces_embeddings = np.load("vectores_rostros_4personas.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

cap = cv.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

    door_triggered = False

    for x, y, w, h in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160,160))
        img = np.expand_dims(img, axis=0)
        ypred = facenet.embeddings(img)
        face_name = model.predict(ypred)
        final_name = encoder.inverse_transform(face_name)[0]

        # Dibujar rectángulo y nombre
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)
        cv.putText(frame, str(final_name), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0,0,255), 2, cv.LINE_AA)

        # Si el rostro es válido → abrir puerta
        if final_name != "desconocido":
            if arduino is not None:
                arduino.write(b'open\n')
                print(">>> Enviado: open")
            door_triggered = True

    # Si no hay rostros válidos → cerrar puerta
    if not door_triggered and arduino is not None:
        arduino.write(b'close\n')
        print(">>> Enviado: close")

    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
if arduino is not None:
    arduino.close()
    print("🔌 Conexión con Arduino cerrada")
