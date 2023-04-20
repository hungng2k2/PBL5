from flask import Flask, render_template, Response
import cv2
import time
from camera import Camera
import socket

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(device):
    """Video streaming generator function."""
    unique_name = device

    num_frames = 0
    total_time = 0
    while True:
        time_start = time.time()

        cam_id, frame = Camera.get_frame(unique_name)
        if frame is None:
            break

        num_frames += 1

        time_now = time.time()
        total_time += time_now - time_start
        fps = num_frames / total_time

        # write camera name
        cv2.putText(frame, cam_id, (int(20), int(
            20 * 5e-3 * frame.shape[0])), 0, 2e-3 * frame.shape[0], (255, 255, 255), 2)

        # Remove this line for test camera
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed/<device>')
def video_feed(device):
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(device),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    hostname = socket.gethostname()
    print(hostname)
    local_ip = socket.gethostbyname(hostname)
    print(local_ip)
    Camera("jetsonnano", 5555)
    try:
        app.run(host='0.0.0.0', threaded=True)
    finally:
        Camera.stop()
