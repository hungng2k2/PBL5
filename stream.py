import imagezmq
import argparse
import socket
import time
from ultralytics import YOLO
import cv2 as cv
import numpy as np
from keras_facenet import FaceNet
import pickle
# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--server-ip", required=True,
                    help="ip address of the server to which the client will connect")
parser.add_argument("-vs", "--video-source", default=0)

args = parser.parse_args()

model_facemask = YOLO('facemaskv0.pt')


def get_embedding(face_pixels, model_facereg=FaceNet()):
    face_pixels = face_pixels.astype('float32')
    samples = np.expand_dims(face_pixels, 0)
    y_hat = model_facereg.embeddings(samples)
    return y_hat[0]


with open('faces_svm.pkl', 'rb') as f:
    model_facereg = pickle.load(f)

with open('output_encoder.pkl', 'rb') as f:
    output_encoder = pickle.load(f)

# initialize the ImageSender object with the socket address of the
# server
sender = imagezmq.ImageSender(connect_to=f"tcp://{args.server_ip}:5555")

# get the host name, initialize the video stream, and allow the
# camera sensor to warmup
hostname = socket.gethostname()

video_source = args.video_source
if isinstance(video_source, str) and video_source.isdigit():
    video_source = int(video_source)
cap = cv.VideoCapture(video_source)

while True:
    # read the frame from the camera and send it to the server
    ret, frame = cap.read()
    if not ret:
        break

    results = model_facemask.predict(source=frame)
    for r in results:

        boxes = r.boxes
        for box in boxes:

            b = box.xyxy[0]
            xmin, ymin, xmax, ymax = list(map(int, b))
            c = box.cls

            cv.rectangle(frame, (xmin, ymin), (xmax, ymax),
                         (0, 255, 0), thickness=2)
            if int(c) == 2:
                pixels = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                xmin = 0 if xmin < 0 else xmin
                ymin = 0 if ymin < 0 else ymin
                face = pixels[ymin:ymax, xmin:xmax]
                face = cv.resize(face, (160, 160))
                face_emb = np.expand_dims(get_embedding(face), 0)
                probs = model_facereg.predict_proba(face_emb)
                prob_max = int(np.amax(probs)*100)/100
                if prob_max < 0.6:
                    predict_name = ['unknown']
                else:
                    y_hat = np.expand_dims(np.argmax(probs), 0)
                    predict_name = output_encoder.inverse_transform(y_hat)
                if predict_name != None:
                    cv.putText(
                        frame, f'{predict_name[0]} {prob_max}', (xmin, ymin), cv.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)

            else:

                confidence = float(box.conf[0])
                cv.putText(frame, f"{model_facemask.names[int(c)]} {round(confidence,2)}", (xmin, ymin),
                           cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)

    reply = sender.send_image(hostname, frame)
    if reply != b'OK':
        break
