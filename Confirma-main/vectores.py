import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder

# ====== Inicializar FaceNet y modelos ======
facenet = FaceNet()
faces_embeddings = np.load("vectores_rostros_4personas.npz")
Y = faces_embeddings['arr_1']

encoder = LabelEncoder()
encoder.fit(Y)

# Cargar Haarcascade y modelo SVM
haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# ====== Iniciar cámara ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    exit()

print("✅ Cámara lista. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de rostros
    faces = haarcascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        face = rgb_img[y:y+h, x:x+w]
        try:
            # Procesar rostro para FaceNet
            face_resized = cv2.resize(face, (160, 160))
            face_resized = np.expand_dims(face_resized, axis=0)

            # Obtener embeddings (vector de características)
            embeddings = facenet.embeddings(face_resized)

            # Mostrar en consola los vectores
            print("\n🔹 Vectores de características:")
            print(embeddings[0])  # un vector de 512 dimensiones

            # Clasificación con SVM
            pred = model.predict(embeddings)
            name = encoder.inverse_transform(pred)[0]

        except Exception as e:
            print("Error procesando rostro:", e)
            name = "desconocido"

        # Dibujar rectángulo y nombre
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(frame, f"{name}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar en ventana
    cv2.imshow("Reconocimiento Facial", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
