"""
MIT License
Copyright (c) 2016 Joshua Goldfarb

Code for determining location of an object in a 2D co-ordinate system using a camera/s

If we get freezing when network cameras drop out: https://github.com/opencv/opencv/issues/4506

http://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
"""
import numpy as np
import cv2
from videoStream import VideoStream
import perspectiveTransformationTest
import logging
import imutils




class Camera:
    """Defines a single camerea"""

    # --- Class Variables ---
    logger = logging.getLogger("Camera")

    # Upper and Lower boundaries for color detection

    # Hue range for red
    hueLower = (170, 70, 50)
    hueUpper = (180, 255, 255)
    hueLower2 = (0, 70, 50)
    hueUpper2 = (10, 255, 255)

    # Hue range for green
    # hueLower = (29, 86, 6)
    # hueUpper = (64, 255, 255)

    # Source Type Enum
    class SourceType:
        USB = "usb"
        STATIC = "static"
        IP = "ip"
        PiCamera = "picamera"

    # ---   ---   ---   ---

    def __init__(self, camera_name, camera_source, source_type, size=None, fps=None):
        """
        Constructor for Camera

        :param camera_name: Used for display purposes only
        :type camera_name: string

        :param camera_source: The video source
        :type camera_source: int or string

        :param source_type: The type of source the video is coming from. (usb, ip, static)
        :type source_type: SourceType

        :param size: Overrides the default resolution of the camera. (width, height)
        :type size: (int, int)
        :param fps: Overrides the default FPS of the camera.
        :type fps: int
        """
        self.camera_name = camera_name

        # open the correct video stream based on the source type
        if source_type == self.SourceType.USB:
            self.video_stream = VideoStream(src=camera_source).start()

        elif source_type == self.SourceType.IP:
            # self.video_stream = VideoStream(src=camera_source, useIPCamera=True).start()
            raise NotImplementedError("IP sources are not yet supported.")

        elif source_type == self.SourceType.PiCamera:
            self.video_stream = VideoStream(src=camera_source, usePiCamera=True).start()

        elif source_type == self.SourceType.STATIC:
            self.video_stream = VideoStream(src=camera_source, useStaticImage=True).start()

        else:
            raise ValueError(source_type + " is not a valid source_type argument.")

        # Load first frame into the frame buffer to initialize.
        self.frame = self.video_stream.read()
        self.warped_frame = None
        #
        # if size is not None:
        #     (override_width, override_height) = size
        #     retval = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, override_width)
        #     retval = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, override_height)
        #
        # if fps is not None:
        #     retval = self.cap.set(cv2.CAP_PROP_FPS, fps)
        #

        # self.width = self.stream().get(cv2.CAP_PROP_FRAME_WIDTH)
        # self.height = self.stream().get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.reference_rectangle = None
        self.calibration_table = None

    def add_calibration_point(self, camera_point, world_point):
        raise NotImplementedError

    def current_location(self):
        raise NotImplementedError

    def locate_tracker(self, debug):
        """
        Returns the (x, y) position of the IR tracker in the camera reference plane.
        http://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

        :return: The (x, y) position of the IR tracker in the camera reference plane.
        :rtype: (int, int)
        """

        # tmp_image =
        # tmp_image = cv2.GaussianBlur(self.frame, (11, 11), 0)  # Experiment with this

        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)  # Convert to HSV Color Space. This is temporary for testing using colored objects)

        mask = cv2.inRange(hsv, self.hueLower, self.hueUpper)

        try:
            mask = cv2.inRange(hsv, self.hueLower2, self.hueUpper2) + mask
        except AttributeError:
            pass

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        if debug:
            tmpMask = imutils.resize(mask, width=1000, height=1000)
            cv2.imshow("mask", tmpMask)


        # find contours in the mask and initialize the current (x, y) center of the object
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)

            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            # if radius > 10:
            #     # draw the circle and centroid on the frame,
            #     # then update the list of tracked points
            #     cv2.circle(frame, (int(x), int(y)), int(radius),
            #                (0, 255, 255), 2)
            #     cv2.circle(frame, center, 5, (0, 0, 255), -1)
            if debug:
                cv2.drawContours(self.frame, c, -1, (0, 255, 0), 20)
            return center, radius
        # update the points queue
        cv2.imshow("mask", imutils.resize(mask, width=1000, height=1000))
        cv2.imshow("frame", imutils.resize(self.frame, width=1000, height=1000))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        raise OpenCVError("Could not find tracker!")

        # return (1, 1), 1

    def set_reference_rectangle(self, reference_rectangle):
        """

        :param reference_rectangle: Four points used to correct for perspective shift. [(x, y), (x, y), (x, y), (x, y)]
        :type reference_rectangle: [(int, int), (int, int), (int, int), (int, int)]
        """
        self.reference_rectangle = np.array(reference_rectangle)

    def update(self):
        """
        Updates the frame buffer
        """
        self.frame = self.video_stream.read()

    def get_new_frame(self):
        self.update()
        return self.frame

    def transform(self):
        if self.reference_rectangle is not None:
            self.warped_frame, junk = perspectiveTransformationTest.four_point_transform(self.frame, self.reference_rectangle)
        else:
            logging.warning("Can not transform frame until reference_rectangle has been set!")

    def stop(self):
        self.video_stream.stop()

    def close(self):
        """
        Clean up Camera.
        :return:
        :rtype:
        """
        self.video_stream.stop()
        self.video_stream.close()

    def stream(self):
        return self.video_stream.stream.stream


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
