# PBL5

# Install

pip install -r server/requirements.txt

pip install -r client_camera/requirements.txt

# Run Server

cd server

python app.py => copy ip/hostname

# Run Client camera

cd client_camera

python app.py --mode 2 --server ip/hostname

args:

--mode [number]

0: show GUI

1: send data to server

2: 0 and 1


--video-source [camera id/video url]

--server [ip/hostname]

--port [server port]

--update-interval [minutes to update faces data from firebase]
