import json
import time
import threading
import imagezmq
from firebase import save

try:
    from greenlet import getcurrent as get_ident
except ImportError:
    try:
        from thread import get_ident
    except ImportError:
        from _thread import get_ident


class CameraEvent:
    """An Event-like class that signals all active clients when a new frame is
    available.
    """

    def __init__(self):
        self.events = {}

    def wait(self):
        """Invoked from each client's thread to wait for the next frame."""
        ident = get_ident()
        if ident not in self.events:
            # this is a new client
            # add an entry for it in the self.events dict
            # each entry has two elements, a threading.Event() and a timestamp
            self.events[ident] = [threading.Event(), time.time()]
        return self.events[ident][0].wait()

    def set(self):
        """Invoked by the camera thread when a new frame is available."""
        now = time.time()
        remove = None
        for ident, event in self.events.items():
            if not event[0].isSet():
                # if this client's event is not set, then set it
                # also update the last set timestamp to now
                event[0].set()
                event[1] = now
            else:
                # if the client's event is already set, it means the client
                # did not process a previous frame
                # if the event stays set for more than 5 seconds, then assume
                # the client is gone and remove it
                if now - event[1] > 5:
                    remove = ident
        if remove:
            del self.events[remove]

    def clear(self):
        """Invoked from each client's thread after a frame was processed."""
        self.events[get_ident()][0].clear()


class Camera:
    UPDATE_DATA_DELAY = 1
    threads = {}  # background thread that reads frames from camera
    frame = {}  # current frame is stored here by background thread
    event = {}
    stop = False

    def __init__(self, device, port):
        """Start the background camera thread if it isn't running yet."""
        self.unique_name = device
        Camera.event[self.unique_name] = CameraEvent()

        if self.unique_name not in Camera.threads:
            Camera.threads[self.unique_name] = None
        if Camera.threads[self.unique_name] is None:
            # start background frame thread
            Camera.threads[self.unique_name] = threading.Thread(target=self._thread,
                                                                args=(self.unique_name, port))
            Camera.threads[self.unique_name].start()

            # wait until frames are available
            while self.get_frame(self.unique_name) is None:
                time.sleep(0)

    @classmethod
    def get_frame(cls, unique_name):
        """Return the current camera frame."""
        # wait for a signal from the camera thread
        Camera.event[unique_name].wait()
        Camera.event[unique_name].clear()

        return Camera.frame[unique_name]

    @staticmethod
    def server_frames(image_hub):
        time_now = time.time()
        time_start = time_now
        while True:  # main loop
            if Camera.stop == True:
                break
            time_now = time.time()

            info, frame = image_hub.recv_image()
            # this is needed for the stream to work with REQ/REP pattern
            image_hub.send_reply(b'OK')
            info = json.loads(info)
            camera_id = info['hostname']

            if (time_now - time_start) >= Camera.UPDATE_DATA_DELAY:
                time_start = time_now
                if len(info['labels']) > 0:
                    save(info)

            yield camera_id, frame

    @classmethod
    def server_thread(cls, unique_name, port):
        device = unique_name

        image_hub = imagezmq.ImageHub(open_port=f'tcp://*:{port}')

        frames_iterator = cls.server_frames(image_hub)
        try:
            for camera_id, frame in frames_iterator:
                Camera.frame[unique_name] = camera_id, frame
                Camera.event[unique_name].set()  # send signal to clients
        except Exception as e:
            frames_iterator.close()
            image_hub.zmq_socket.close()
            print(f'Closing server socket at port {port}.')
            print(f'Stopping server thread for device {device} due to error.')
            print(e)

    @classmethod
    def _thread(cls, unique_name, port):
        device = unique_name
        port = port
        print(f'Starting server thread for device {device} at port {port}.')
        cls.server_thread(unique_name, port)

        Camera.threads[unique_name] = None

    @classmethod
    def stop(cls):
        Camera.stop = True
