"""
MIT License
Copyright (c) 2016 Joshua Goldfarb

Code for determining location of an object in a 2D co-ordinate system using a camera/s

If we get freezing when network cameras drop out: https://github.com/opencv/opencv/issues/4506

http://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
"""
import numpy
import cv2


class Camera:
    """Defines a single camerea"""

    def __init__(self, camera_name, usb_camera_number=None, http_camera_url=None, size=None, fps=None):
        """
        Constructor for Camera

        :param camera_name: Used for display purposes only
        :type camera_name: string
        :param usb_camera_number: usb camera number to open. usb_camera_number OR http_camera_url must be defined.
        :type usb_camera_number: int
        :param http_camera_url: URL to web camera. usb_camera_number OR http_camera_url must be defined.
        :type http_camera_url: string
        :param size: Overrides the default resolution of the camera. (width, height)
        :type size: (int, int)
        :param fps: Overrides the default FPS of the camera.
        :type fps: int
        """
        self.camera_name = camera_name
        self.frame = None

        if usb_camera_number is not None:
            # self.cap = cv2.VideoCapture(usb_camera_number)
            self.cap = cv2.VideoCapture()
            self.cap.open(usb_camera_number)
        elif http_camera_url is not None:  # TODO: Combine USB and HTTP initialisation
            self.cap = cv2.VideoCapture()
            self.cap.open(http_camera_url)

        if size is not None:
            (override_width, override_height) = size
            retval = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, override_width)
            retval = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, override_height)

        if fps is not None:
            retval = self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def retrieve(self):
        """
        Decodes the last frame that was grabbed.
        """
        retval, frame = self.cap.retrieve()
        if retval is False:
            raise OpenCVError(message="Failed to decode frame from camera " + self.camera_name)
        else:
            self.frame = frame

    def grab(self):
        """
        Grabs the next frame from the camera but does not decode it.
        """
        if self.cap.isOpened():
            if self.cap.grab() is False:
                raise OpenCVError(message="Failed to grab frame from camera " + self.camera_name)
        else:
            raise OpenCVError(message="Failed to grab frame from camera " + self.camera_name +
                                      ". Camera feed is not open!")

    def close(self):
        """
        Clean up Camera.
        :return:
        :rtype:
        """

        self.cap.release()


class OpenCVError(Exception):
    """"""
    pass
    # def __init__(self, message, errors):
    #     """Constructor for OpenCVError(Exception)"""
    #     # Call the base class constructor with the parameters it needs
    #     super(OpenCVError, self).__init__(message)
    #
    #     # Now for your custom code...
    #     self.errors = errors