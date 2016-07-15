"""
The MIT License (MIT)

Copyright (c) 2015 Adrian Rosebrock, http://www.pyimagesearch.com
"""


# import the necessary packages
from threading import Thread
import cv2


class StaticImageStream:
    """
    Testing class that loads a static img file
    """

    def __init__(self, src=0):

        self.frame = cv2.imread(src, cv2.IMREAD_COLOR)

        # initialize the variable used to indicate if the thread should
        # be stopped

    def start(self):
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        pass

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        pass

    def close(self):
        """
        Clean up OpenCV Camera
        """
        pass
