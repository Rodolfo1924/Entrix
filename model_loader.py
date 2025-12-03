import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelEncoder
import glob

class FaceModel:
    _model = None
    _encoder = None

    @staticmethod
    def load():
        if FaceModel._model is not None:
            return FaceModel._model, FaceModel._encoder

        rostros_path = "entrix/rostros/"
        face_images, face_labels = [], []

        for usuario in os.listdir(rostros_path):
            for path in glob.glob(f"{rostros_path}/{usuario}/*.jpg"):
                img = cv2.imread(path, 0)
                face_images.append(img)
                face_labels.append(usuario)

        encoder = LabelEncoder()
        y = encoder.fit_transform(face_labels)

        model = cv2.face.EigenFaceRecognizer_create()
        if len(face_images) > 0:
            model.train(face_images, np.array(y))

        FaceModel._model = model
        FaceModel._encoder = encoder
        return model, encoder
