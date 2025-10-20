# face_recognition_door.py
import cv2
import numpy as np
import serial
import time
import os
from keras_facenet import FaceNet
import pickle
from sklearn.preprocessing import LabelEncoder

try:
    # Configura Arduino
    try:
        arduino = serial.Serial('COM5', 9600, timeout=1)
        time.sleep(2)
    except Exception as e:
        print(f"Error con Arduino: {e}")
        arduino = None

    # Carga el modelo
    try:
        facenet = FaceNet()
        faces_embeddings = np.load("vectores_rostros_4personas.npz")
        Y = faces_embeddings['arr_1']
        encoder = LabelEncoder()
        encoder.fit(Y)
        haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        model = pickle.load(open("svm_model_160x160.pkl", 'rb'))
    except Exception as e:
        print(f"Error cargando modelos: {e}")
        exit()

    # Inicia cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame")
            break

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
        except Exception as e:
            print(f"Error detectando rostros: {e}")
            break

        door_triggered = False

        for (x, y, w, h) in faces:
            try:
                img = rgb_img[y:y+h, x:x+w]
                img = cv2.resize(img, (160, 160))
                img = np.expand_dims(img, axis=0)
                ypred = facenet.embeddings(img)
                face_name = model.predict(ypred)
                proba = model.predict_proba(ypred)[0].max()
                
                final_name = "desconocido" if proba < 0.7 else encoder.inverse_transform(face_name)[0]
                color = (0, 255, 0) if final_name != "desconocido" else (0, 0, 255)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, final_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.9, color, 2)
    
                if final_name != "desconocido" and arduino is not None:
                    arduino.write(b'open\n')
                    door_triggered = True

            except Exception as e:
                print(f"Error procesando rostro: {e}")
                continue

        if not door_triggered and arduino is not None:
            arduino.write(b'close\n')

        cv2.imshow("Control de Acceso Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Programa interrumpido")

finally:
    if 'arduino' in locals() and arduino is not None:
        arduino.close()
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    print("Recursos liberados")