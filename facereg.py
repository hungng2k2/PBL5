from keras_facenet import FaceNet
import pickle
import os
import numpy as np
import cv2 as cv


class FaceRegconition:
    def __init__(self, model_path, threshold=0.6) -> None:
        self.load(model_path)
        self.threshold = threshold

    def load(self, model_path):
        with open(os.path.join(model_path, 'faces_svm.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(model_path, 'output_encoder.pkl'), 'rb') as f:
            self.output_encoder = pickle.load(f)

    def get_embedding(self, face_pixels, model=FaceNet()):
        face_pixels = face_pixels.astype('float32')
        samples = np.expand_dims(face_pixels, 0)
        y_hat = model.embeddings(samples)
        return y_hat[0]

    def predict(self, face_pixels):
        size = (160, 160)
        face_pixels = cv.resize(face_pixels, size)
        face_emb = np.expand_dims(self.get_embedding(face_pixels), 0)
        probs = self.model.predict_proba(face_emb)
        prob_max = int(np.amax(probs)*100)/100
        if prob_max < 0.6:
            predict_name = ['unknown']
        else:
            y_hat = np.expand_dims(np.argmax(probs), 0)
            predict_name = self.output_encoder.inverse_transform(y_hat)
        return predict_name, prob_max
