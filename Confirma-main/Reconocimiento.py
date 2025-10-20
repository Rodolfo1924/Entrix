import cv2
import numpy as np
import serial
import time
from ultralytics import YOLO
from keras_facenet import FaceNet
import pickle
from sklearn.preprocessing import LabelEncoder


MIN_MOVEMENT_THRESHOLD = 2.0 
FRAMES_TO_ANALYZE = 5  

try:
    #Arduino
    try:
        arduino = serial.Serial('COM5', 9600, timeout=1)
        time.sleep(2)
    except Exception as e:
        print(f"Error con Arduino: {e}")
        arduino = None

    # modelos
    try:
        # Modelo de rostros
        face_detector = YOLO('weights\yolov8n-face.pt') 
        
        
        facenet = FaceNet()
        faces_embeddings = np.load("vectores_rostros_4personas.npz")
        Y = faces_embeddings['arr_1']
        encoder = LabelEncoder()
        encoder.fit(Y)
        model = pickle.load(open("svm_model_160x160.pkl", 'rb'))
    except Exception as e:
        print(f"Error cargando modelos: {e}")
        exit()

    #  cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        exit()

    prev_gray = None
    movement_history = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame")
            break

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        results = face_detector(frame, verbose=False)
        faces = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                faces.append((x1, y1, x2-x1, y2-y1))  # Convertir a formato (x,y,w,h)

        door_triggered = False
        movement_detected = False

        # Calcula el flujo óptico si hay rostros detectados
        if len(faces) > 0 and prev_gray is not None:
            # Usamos el primer rostro detectado para el análisis de movimiento
            (x, y, w, h) = faces[0]
            face_roi = prev_gray[y:y+h, x:x+w]
            
            # Preparamos puntos para el flujo óptico
            p0 = cv2.goodFeaturesToTrack(face_roi, maxCorners=50, 
                                       qualityLevel=0.3, minDistance=7, blockSize=7)
            
            if p0 is not None:
                # Calcula el flujo óptico
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray_img, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # Calcula el movimiento promedio
                if flow is not None:
                    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                    movement = np.mean(magnitude[y:y+h, x:x+w])  # Solo en la región del rostro
                    movement_history.append(movement)
                    
                    # Mantiene solo los últimos N movimientos
                    if len(movement_history) > FRAMES_TO_ANALYZE:
                        movement_history.pop(0)
                    
                    # Determina si hay suficiente movimiento
                    if len(movement_history) == FRAMES_TO_ANALYZE:
                        avg_movement = np.mean(movement_history)
                        movement_detected = avg_movement > MIN_MOVEMENT_THRESHOLD
                        cv2.putText(frame, f"Movimiento: {avg_movement:.2f}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f" {'' if movement_detected else 'Analizando'}", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if movement_detected else (0, 0, 255), 2)

        prev_gray = gray_img.copy()
        frame_count += 1

        for (x, y, w, h) in faces:
            try:
                img = rgb_img[y:y+h, x:x+w]
                img = cv2.resize(img, (160, 160))
                img = np.expand_dims(img, axis=0)
                ypred = facenet.embeddings(img)
                face_name = model.predict(ypred)
                proba = model.predict_proba(ypred)[0].max()
                
                final_name = "desconocido" if proba < 0.7 else encoder.inverse_transform(face_name)[0]
                
                # Solo abre si hay movimiento detectado (anti-spoofing)
                if final_name != "desconocido" and (frame_count <= FRAMES_TO_ANALYZE or movement_detected):
                    color = (0, 255, 0)  # Verde - reconocido y vivo
                    if arduino is not None:
                        arduino.write(b'open\n')
                        door_triggered = True
                else:
                    color = (0, 120, 255)  # Naranja - reconocido pero posible spoof
                
                if final_name == "desconocido":
                    color = (0, 0, 255)  # Rojo - desconocido
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, final_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.9, color, 2)

            except Exception as e:
                print(f"Error procesando rostro: {e}")
                continue

        if not door_triggered and arduino is not None:
            arduino.write(b'close\n')

        cv2.imshow("Control de Acceso Facial - YOLOv8", frame)
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