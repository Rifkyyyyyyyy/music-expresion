# utils/webcam.py

import cv2
from threading import Thread

class WebcamVideoStream:
    """
    Class for accessing webcam video stream in a separate thread to improve performance.
    """

    def __init__(self, src=0):
        # Initialize video capture with given source (default is webcam 0)
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            raise RuntimeError("Unable to open video source", src)
        
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        """Start the background thread to read frames from the video stream."""
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        """Continuously capture frames until the stream is stopped."""
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()
            if not self.grabbed:
                self.stop()
                break

    def read(self):
        """Return the most recent frame."""
        return self.frame

    def stop(self):
        """Stop the video stream."""
        self.stopped = True
        self.stream.release()
