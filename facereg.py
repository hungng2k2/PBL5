from keras_facenet import FaceNet
import pickle
import os
import numpy as np
import cv2 as cv
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans


import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
    
class FaceNet_Recognizer_EuclideanDistance():
    def __init__(self, n_clusters=5):
        self.UNKNOWN_LABEL = 'unknown'
        self.image_size = (160, 160)
        self.threshold = 1
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.cluster_centers = np.empty((0,512))
        self.labels = np.empty((0))
        self.last_update = 0

    def __str__(self):
        return f"""
            n_clusters = {self.n_clusters}
            threshold = {self.threshold}
            unique_labels_num = {np.unique(self.labels).size}
            unique_labels = {np.unique(self.labels)}
            """

    def load(self, model_path):
        with open(os.path.join(model_path, 'FaceNet_Recognizer_EuclideanDistance.pkl'), 'rb') as file:
            data = pickle.load(file)
        self.threshold = data['threshold']
        self.n_clusters = data['n_clusters']
        self.kmeans = KMeans(n_clusters=data['n_clusters'])
        self.cluster_centers = data['cluster_centers']
        self.labels = data['labels']

    def load_from_firebase(self):
        try:
            # Retrieving model data
            model_data_ref = db.collection('model_data')
            query = model_data_ref.order_by(
                u'time', direction=firestore.Query.DESCENDING).limit(1)
            result = query.get()
            model_data = result[0].to_dict()
            # Check if firebase is updated
            if self.last_update < model_data['time']:
                self.last_update = model_data['time']
                self.n_clusters = model_data['n_clusters']
                self.kmeans = KMeans(n_clusters=model_data['n_clusters'])
                self.threshold = model_data['threshold']
                # Retrieving face data
                faces = db.collection('face').stream()
                for face in faces:
                    for vector in face.to_dict().values():
                        self.cluster_centers = np.append(self.cluster_centers, [vector], axis=0)
                        self.labels = np.append(self.labels, [face.id], axis=0)
        except Exception as e:
            print(e)

    def get_embedding(self, face_pixels, model=FaceNet()):
        '''
        Input: cropped face
        Output: embedding vector
        '''
        # chuyển thành float
        face_pixels = face_pixels.astype('float32')
        # chuyển thành dạng tensor [[1,2,3,12,3]]
        samples = np.expand_dims(face_pixels, 0)
        # trích xuất ma trận đặc trưng
        y_hat = model.embeddings(samples)
        return y_hat[0]
        
    def predict(self, X_test):
        X_test_emb = np.array(list(map(lambda x: self.get_embedding(x), X_test)))
        y_predict = []
        for x in X_test_emb:
            distances = euclidean_distances(x.reshape(1, -1), self.cluster_centers)
            if np.min(distances) > self.threshold + 0.4:
                y_predict.append(self.UNKNOWN_LABEL)
            else:
                min_index = np.argmin(distances)
                y_predict.append(self.labels[min_index])
        return np.array(y_predict)
        
    def predict_from_image(self, face_pixels):
        face_pixels = cv.resize(face_pixels, self.image_size)      
        return self.predict([face_pixels])
