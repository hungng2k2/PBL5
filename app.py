import time
import imagezmq
import socket
from ultralytics import YOLO
import cv2 as cv
import argparse
from facereg import FaceRecognition
import json

parser = argparse.ArgumentParser(prog='facemask')
parser.add_argument('-v', '--video-source', default=0,
                    help="camera id or video streaming url")
parser.add_argument('-m', '--mode', default=0,
                    help="0: show GUI, 1: stream to server, 2: both")
parser.add_argument("-s", "--server", default=None,
                    help="ip address or host name of the server to which the client will connect")
parser.add_argument("-p", "--port", default=5555,
                    help="server port to stream video")
parser.add_argument("--facereg-path", default="facereg-v0",
                    help="face regconition model directory")
parser.add_argument("--facemask-path", default="facemaskv0.pt",
                    help="facemask YOLO model")

args = parser.parse_args()

video_source = args.video_source
server = args.server
port = args.port
facereg_path = args.facereg_path
facemask_path = args.facemask_path

# mode
# 0: show GUI
# 1: stream to server
# 2: both
mode = int(args.mode)


if (mode == 1 or mode == 2) and server == None:
    raise RuntimeError("server is none")

# load YOLO facemask model
model_facemask = YOLO(facemask_path)
# load face regconition model
model_facereg = FaceRecognition(facereg_path, threshold=0.5)

if mode == 1 or mode == 2:
    # initialize the ImageSender object with the socket address of the
    # server
    sender = imagezmq.ImageSender(connect_to=f"tcp://{server}:{port}")
    # get the host name
    hostname = socket.gethostname()

if isinstance(video_source, str) and video_source.isdigit():
    video_source = int(video_source)

cap = cv.VideoCapture()
cap.open(video_source)

while True:

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.flip(frame, 1)

    time_start = int(time.time() * 1000)

    labels = []
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
                predict_name, prob_max = model_facereg.predict(face)
                if predict_name != None:
                    label = predict_name[0]
                    cv.putText(
                        frame, f'{label} {prob_max}', (xmin, ymin), cv.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
            else:
                confidence = float(box.conf[0])
                label = model_facemask.names[int(c)]
                cv.putText(frame, f"{label} {round(confidence,2)}", (xmin, ymin),
                           cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)

            labels.append(label)

    if mode == 0 or mode == 2:
        cv.imshow('facemask', frame)
    if mode == 1 or mode == 2:
        info = {
            "hostname": hostname,
            "time": time.ctime(),
            "labels": labels
        }
        reply = sender.send_image(json.dumps(info), frame)
        if reply != b'OK':
            print("Server not response")
            if mode == 2:
                mode = 0
    
    time_end = int(time.time() * 1000)

    if cv.waitKey(time_end - time_start) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
