from imutils import build_montages
from datetime import datetime
import imagezmq
import argparse
import cv2
import socket
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-mW", "--montageW", type=int, default=1,
                help="montage frame width")
ap.add_argument("-mH", "--montageH", type=int, default=1,
                help="montage frame height")
args = ap.parse_args()

# initialize the ImageHub object
imageHub = imagezmq.ImageHub()

frameDict = {}

# initialize the dictionary which will contain  information regarding
# when a device was last active, then store the last time the check
# was made was now
lastActive = {}
lastActiveCheck = datetime.now()

ESTIMATED_NUM_PIS = 4
ACTIVE_CHECK_PERIOD = 10
ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

mW = args.montageW
mH = args.montageH

hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
print(local_ip)

# start looping over all the frames
while True:
    (clientname, frame) = imageHub.recv_image()
    (h, w) = frame.shape[:2]
    # if a device is not in the last active dictionary then it means
    # that its a newly connected device
    if clientname not in lastActive.keys():
        print(f"[INFO] receiving data from {clientname}...")
    # record the last active time for the device from which we just
    # received a frame
    lastActive[clientname] = datetime.now()

    # update the new frame in the frame dictionary
    frameDict[clientname] = frame
    # build a montage using images in the frame dictionary
    montages = build_montages(frameDict.values(), (w, h), (mW, mH))
    # display the montage(s) on the screen
    for (i, montage) in enumerate(montages):
        cv2.imshow(f"monitor ({i})", montage)
    # detect any kepresses
    key = cv2.waitKey(1) & 0xFF

    # if current time *minus* last time when the active device check
    # was made is greater than the threshold set then do a check
    if (datetime.now() - lastActiveCheck).seconds > ACTIVE_CHECK_SECONDS:
        # loop over all previously active devices
        for (clientname, ts) in list(lastActive.items()):
            # remove the device from the last active and frame
            # dictionaries if the device hasn't been active recently
            if (datetime.now() - ts).seconds > ACTIVE_CHECK_SECONDS:
                print(f"[INFO] lost connection to {clientname}")
                lastActive.pop(clientname)
                frameDict.pop(clientname)
        lastActiveCheck = datetime.now()

    if key == ord("q"):
        imageHub.send_reply(b'F')
        break
    imageHub.send_reply(b'OK')


cv2.destroyAllWindows()
