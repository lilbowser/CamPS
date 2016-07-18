"""
MIT License

Copyright (c) 2016 Joshua Goldfarb

"""
import cv2
import math

from matplotlib._path import points_in_path

platformSize = (300, 400, 200)  # width, length, and height of motion platform
markerPosition = (0, 0, 101)  # X, Y, Z position of marker from center of motion platform

imageSize = (630, 630)

coordinateMap = [[None for x in range(imageSize[0])] for y in range(imageSize[1])]

coordinateMap[172][43]  = [250, 250]
coordinateMap[129][495] = [250, 1250]
coordinateMap[258][559] = [1250, 1250]
coordinateMap[602][323] = [1250, 250]


class InterpolationMap:
    """
    Data structure storing map to deskew image to match real life
    """
    def __init__(self):

        self.list = LinkedList()

    def add_point(self, camera_x, camera_y, world_x, world_y):
        """
        Creates a new lookup point
        :param camera_x: refers to the point on the camera
        :type camera_x: float
        :param camera_y: refers to the point on the camera
        :type camera_y: float
        :param world_x: refers to RL coord
        :type world_x: float
        :param world_y: refers to RL coord
        :type world_y: float
        :return:
        :rtype:
        """
        self.list.append(InterpolationData(camera_x, camera_y, world_x, world_y))

    def get_world_coord_from(self, camera_x, camera_y):
        raise NotImplementedError

    def get_adjacent_data(self, camera_position):
        raise NotImplementedError

    def get_closest_points(self, point, n=1):
        return self.list.closest_camera_points(point, n)

    def get_point_pairs(self, points, target):
        """

        :param points:
        :type points: list[list[float|Node]]
        :param target:
        :type target: Point
        :return:
        :rtype:
        """

        default_entry = (999999999999, None, None)
        closest_found = []
        for i in range(len(points)):
            closest_found.append(default_entry)

        for i, entry in enumerate(points):
            point = entry[1].data.camera_position()
            for j, entry2 in enumerate(points):
                if i != j:
                    point2 = entry2[1].data.camera_position()

                    # Find the distance from the point to the line
                    denom = math.sqrt((point2.x - point.x) ** 2 + (point2.y - point.y) ** 2)
                    numer = abs(((point2.y - point.y) * target.x) - ((point2.x - point.x) * target.y) + (point2.x * point.y) - (point2.y * point.x))
                    line_distance = numer / denom

                    if line_distance < closest_found[0][0]:
                        duplicate = False
                        for data in closest_found:
                            if data[0] == line_distance:
                                duplicate = True

                        if not duplicate:
                            closest_found[0] = (line_distance, point, point2)
                            closest_found = sorted(closest_found, key=lambda tup: tup[0], reverse=True) #sorted(closest_found, reverse=True)

        while default_entry in closest_found:
            closest_found.remove(default_entry)

        return closest_found


class LinkedList(object):

    def __init__(self, head=None):
        """

        :param head:
        :type head: Node
        """
        self.head = head

    def append(self, data):
        """

        :param data: Data entry containing Deskew Information
        :type data: InterpolationData
        :return:
        :rtype:
        """
        new_node = Node(data)
        new_node.set_next(self.head)
        self.head = new_node

    def size(self):
        current = self.head
        count = 0
        while current:
            count += 1
            current = current.get_next()
        return count

    def search(self, data):
        current = self.head
        found = False
        while current and found is False:
            if current.get_data() == data:
                found = True
            else:
                current = current.get_next()
        if current is None:
            raise ValueError("Data not in list")
        return current

    def search_camera(self, camera_position):
        current = self.head
        found = False
        while current and found is False:
            if current.get_data().camera_position() == camera_position:
                found = True
            else:
                current = current.get_next()
        if current is None:
            raise ValueError("Data not in list")
        return current

    def closest_camera_points(self, search_point, n=1):
        """

        :param search_point:
        :type search_point: Point
        :return:
        :rtype:
        """

        # TODO: This probably needs to be optimised
        current = self.head

        closest_found = []
        for i in range(n):
            closest_found.append((999999999999, None))

        if current is None:
            raise ValueError("Data not in list")

        while current:  # and number_found < n:
            distance = current.get_data().distance_to_camera_point(search_point)

            if distance < closest_found[0][0]:
                closest_found[0] = (distance, current)
                closest_found = sorted(closest_found, key=lambda tup: tup[0], reverse=True) # sorted(closest_found, reverse=True)

            current = current.get_next()

        return closest_found

    def delete(self, data):
        current = self.head
        previous = None
        found = False
        while current and found is False:
            if current.get_data() == data:
                found = True
            else:
                previous = current
                current = current.get_next()
        if current is None:
            raise ValueError("Data not in list")
        if previous is None:
            self.head = current.get_next()
        else:
            previous.set_next(current.get_next())


class Node:

    def __init__(self, data, next_node=None):
        """

        :param data: Data entry containing Deskew Information
        :type data: InterpolationData
        :param next_node:
        :type next_node:
        """
        self.data = data
        self.next_node = next_node

    def get_data(self):
        """

        :return: InterpolationData
        :rtype: InterpolationData
        """
        return self.data

    def get_next(self):
        return self.next_node

    def set_next(self, new_next):
        self.next_node = new_next

class InterpolationData:
    # Create better name

    def __init__(self, camera_x, camera_y, world_x, world_y):
        self.camera_x = camera_x
        self.camera_y = camera_y
        self.world_x = world_x
        self.world_y = world_y

    def difference(self):
        return Point(self.camera_x - self.world_x, self.camera_y - self.world_y)

    def camera_position(self):
        return Point(self.camera_x, self.camera_y)

    def world_position(self):
        return Point(self.world_x, self.world_y)

    def distance_to_camera_point(self, point):
        """
        Returns the distance from given coordinates to the stored camera point
        :param point: The point we want distance from.
        :type point: Point
        :return:
        :rtype: double
        """
        return math.sqrt((point.x - self.camera_x)**2 + (point.y - self.camera_y)**2)

class InterpolationDataSingle:
    # Create better name

    # def __init__(self, primary_x, primary_y, sub_x, sub_y):
    #     self.primary_x = primary_x
    #     self.primary_y = primary_y
    #     self.sub_x = sub_x
    #     self.sub_y = sub_y
    #
    # def difference(self):
    #     return Point(self.primary_x - self.sub_x, self.primary_y - self.sub_y)


    def __init__(self, camera, world):
        """

        :param camera: Coord from camera
        :type camera: coord from
        :param world:
        :type world:
        """
        self.camera = camera
        self.world = world

    def difference(self):
        return self.camera - self.world


class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

