
import cv2
platformSize = (300, 400, 200)  # width, length, and height of motion platform
markerPosition = (0, 0, 101)  # X, Y, Z position of marker from center of motion platform

imageSize = (640, 640)

coordinateMap = [[None for x in range(imageSize[0])] for y in range(imageSize[1])]

coordinateMap[172][43]  = [250, 250]
coordinateMap[129][495] = [250, 1250]
coordinateMap[258][559] = [1250, 1250]
coordinateMap[602][323] = [1250, 250]


class DeskewMap:
    """
    Data structure storing map to deskew image to match real life
    """
    def __init__(self):
        self.x_list = LinkedList()
        self.y_list = LinkedList()
        self.list = LinkedList()#[]

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
        # self.x_list.append(DeskewData(camera_x, world_x))
        # self.y_list.append(DeskewData(camera_y, world_y))

        self.x_list.append(DeskewData(camera_x, camera_y, world_x, world_y))

    def get_world_coord_from(self, camera_x, camera_Y):
        raise NotImplementedError

    def get_adjacent_data(self, camera_position):



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
        :type data: DeskewData
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
            if current.get_data().camera == data:
                found = True
            else:
                current = current.get_next()
        if current is None:
            raise ValueError("Data not in list")
        return current

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
        :type data: DeskewData
        :param next_node:
        :type next_node:
        """
        self.data = data
        self.next_node = next_node

    def get_data(self):
        """

        :return: DeskewData
        :rtype: DeskewData
        """
        return self.data

    def get_next(self):
        return self.next_node

    def set_next(self, new_next):
        self.next_node = new_next

class DeskewData:
    # Create better name

    def __init__(self, camera_x, camera_y, world_x, world_y):
        self.camera_x = camera_x
        self.camera_y = camera_y
        self.world_x = world_x
        self.world_y = world_y

    def difference(self):
        return Point(self.camera_x - self.world_x, self.camera_y - self.world_y)

class DeskewDataSingle:
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

