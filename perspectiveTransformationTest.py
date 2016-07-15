"""
MIT License

Copyright (c) 2016 Joshua Goldfarb

Perspective Transformation testing
cCode used from http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

"""
import cv2
# import cv
import numpy as np
import math
import copy

from imutils import perspective
from imutils import contours
import imutils
# from scipy.spatial import distance as dist


def order_points(pts):
    return imutils.perspective.order_points(pts)
    # # sort the points based on their x-coordinates
    # xSorted = pts[np.argsort(pts[:, 0]), :]
    #
    # # grab the left-most and right-most points from the sorted
    # # x-roodinate points
    # leftMost = xSorted[:2, :]
    # rightMost = xSorted[2:, :]
    #
    # # now, sort the left-most coordinates according to their
    # # y-coordinates so we can grab the top-left and bottom-left
    # # points, respectively
    # leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    # (tl, bl) = leftMost
    #
    # # now that we have the top-left coordinate, use it as an
    # # anchor to calculate the Euclidean distance between the
    # # top-left and right-most points; by the Pythagorean
    # # theorem, the point with the largest distance will be
    # # our bottom-right point
    # D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    # (br, tr) = rightMost[np.argsort(D)[::-1], :]
    #
    # # return the coordinates in top-left, top-right,
    # # bottom-right, and bottom-left order
    # return np.array([tl, tr, br, bl], dtype="float32")


def order_points_old(pts):
    """

    :param pts: list of 4 (x,y) points
    :type pts:
    :return:
    :rtype:
    """
    # initialise a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def mult(m, p):

    result = np.array([
                m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
                m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
                m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2]
            ])
    return result


def add(m1, m2):
    result = np.array((m1[0].T + m2[0].T, m1[1].T + m2[1].T))
    # result = np.array((m1[0] + m2[0], m1[1] + m2[1]))
    return result.flatten()


def four_point_transform(image, pts):
    """

    :param image:
    :type image:
    :param pts:
    :type pts:
    :return:
    :rtype:
    """
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    rect_o = order_points_old(pts)
    (tl, tr, br, bl) = rect_o

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # dst = np.array([
    #     [78, 279],
    #     [maxWidth - 1, 279],
    #     [maxWidth - 1, maxHeight - 1],
    #     [78, maxHeight - 1]], dtype="float32")
    #
    # dst = np.array([
    #     [0, 0],
    #     [530, 0],
    #     [530, 615],
    #     [0, 615]], dtype="float32")

    # dst = np.array([
    #     [0+78, 0+279],
    #     [530+78, 0+279],
    #     [530+78, 615+279],
    #     [0+78, 615+279]], dtype="float32")
    # dst = np.array([
    #     [-10-1, -10-1],
    #     [-10 - 1, 0],
    #     [100 - 1, 100 - 1],
    #     [0, 100 - 1]], dtype="float32")
    # dst = np.array([
    #     [0, 0],
    #     [50, 0],
    #     [50, 50],
    #     [0, 50]], dtype="float32")
    # dst = np.zeros((3000, 3000, 3), np.uint8)

    # compute the perspective transform matrix and then apply it
    # M = cv2.getPerspectiveTransform(rect, dst)
    M = cv2.getPerspectiveTransform(rect, dst)  # Get the initial Perspective Transformation Matrix

    # origHeight, origWidth, channels = image.shape  # Get the size of the original image
    # pts = np.float32([[0, 0], [0, origHeight], [origWidth, origHeight], [origWidth, 0]]).reshape(-1, 1, 2)
    #
    # [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    # [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    # t = [-xmin, -ymin]
    # Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate
    # warped = cv2.warpPerspective(image, Ht.dot(M), (xmax-xmin, ymax-ymin))#, cv2.WARP_INVERSE_MAP)  # (maxWidth+100, maxHeight+100))

    # warped = cv2.warpPerspective(image, M, (3000, 3000))#(maxWidth+100, maxHeight+100))
    # warped = cv2.warpPerspective(image, M, (1500, 1500), cv2.WARP_INVERSE_MAP)  # (maxWidth+100, maxHeight+100))
    # return the warped image

    # http://stackoverflow.com/questions/19695702/opencv-wrapperspective-on-whole-image
    rows, columns, channels = image.shape  # Get the size of the original image
    # a = cv2.transform(np.linalg.inv(M), (0, 0, 1))

    a_column = np.array([0, 0, 1])
    # a = np.linalg.inv(M) * a_column#(0, 0, 1)
    a = mult(np.linalg.inv(M), a_column)
    a = a*(1.0/a[2])

    b_column = np.array([0, rows, 1])
    # b = np.linalg.inv(M) * b_column#(0, rows, 1)
    pb = mult(np.linalg.inv(M), b_column)
    b = pb * (1.0 / pb[2])

    c_column = np.array([columns, rows, 1])
    # c_column = np.array([[columns], [rows], [1]])
    # c = np.linalg.inv(M) * c_column#(columns, rows, 1)
    c = mult(np.linalg.inv(M), c_column)
    c = c * (1.0 / c[2])

    d_column = np.array([columns, 0, 1])
    # d_column = np.array([[columns], [0], [1]])
    # d = np.linalg.inv(M) * d_column#(columns, 0, 1)
    d = mult(np.linalg.inv(M), d_column)
    d = d * (1.0/d[2])

    x1 = np.absolute(np.minimum(np.minimum(a[0], b[0]), np.minimum(c[0], d[0])))
    x = np.ceil(x1)
    y1 = np.absolute(np.minimum(np.minimum(a[1], b[1]), np.minimum(c[1], d[1])))
    y = np.ceil(y1)

    x1 = np.absolute(np.maximum(np.maximum(a[0], b[0]), np.maximum(c[0], d[0])))
    x = np.ceil(x1)
    y1 = np.absolute(np.maximum(np.maximum(a[1], b[1]), np.maximum(c[1], d[1])))
    y = np.ceil(y1)

    width = np.ceil(np.absolute(np.maximum(np.maximum(a[0], b[0]), np.maximum(c[0], d[0])))) + x
    height = np.ceil(np.absolute(np.maximum(np.maximum(a[1], b[1]), np.maximum(c[1], d[1])))) + y

    add_array = (x*4, y*25)
    new_rect = copy.deepcopy(rect)
    new_dst = copy.deepcopy(dst)
    for i in range(0, 4):
        # rect[i] += [x, y]
        new_rect[i] = np.add(new_rect[i], add_array)
        new_dst[i] = np.add(new_dst[i], add_array)
    # tmp = add(rect[0], add_array)
    new_M = cv2.getPerspectiveTransform(rect, new_dst)
    # MM = cv2.getPerspectiveTransform(rect, dst)

    # warped_full = cv2.warpPerspective(image, new_M, (width.astype(np.int64), height.astype(np.int64)))#(maxWidth, maxHeight))  # (3000, 3000), image)#
    # warped_part = cv2.warpPerspective(image, new_M, (width.astype(np.int64), height.astype(np.int64)))#, cv2.WARP_INVERSE_MAP)#(maxWidth, maxHeight))  # (3000, 3000), image)#
    #
    # warped_full = cv2.warpPerspective(image, new_M, (5000, 5000), flags= cv2.WARP_INVERSE_MAP)#, cv2.WARP_INVERSE_MAP)
    # warped_part = cv2.warpPerspective(image, M, (1000, 1000), flags= cv2.WARP_INVERSE_MAP)#(width, height))

    warped_full = cv2.warpPerspective(image, new_M, (10000, 15000))#, flags=cv2.WARP_INVERSE_MAP)  # , cv2.WARP_INVERSE_MAP)
    warped_part = cv2.warpPerspective(image, M, (1000, 1000))#, flags=cv2.WARP_INVERSE_MAP)  # (width, height))
    return warped_full, warped_part


def get_cropped_img(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (7, 7), 0)

    edged = cv2.Canny(grey, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # cv2.imshow('edged', edged)
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    largest_contour_area = 1000000000
    largest_contour = None
    for (i, c) in enumerate(cnts):
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue
        if cv2.contourArea(c) < largest_contourArea:
            largest_contour_area = cv2.contourArea(c)
            largest_contour = c

        # bounding_box = cv2.boundingRect(c)
        # x1, y1, h, w = bounding_box
        # cv2.drawContours(image, c, -1, (0, 255, 0), 20)
        # # cv2.rectangle(image, (x1, y1), (x1 + h, y1 + w), (0, 255, 0), 2)
        # # image = image[bounding_box[2]:bounding_box[0], bounding_box[3]:bounding_box[1]]

    if largest_contour is not None:
        bounding_box = cv2.boundingRect(largest_contour)
        x1, y1, h, w = bounding_box
        # cv2.drawContours(image, bounding_box, -1, (0, 255, 0), -1)
        cv2.rectangle(image, (x1, y1), (x1+ h, y1+w), (255, 255, 0), 5)
        image = image[y1:y1+w, x1:x1+h]
        return image
    return None


def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image

if __name__ == '__main__':
    # img = cv2.imread("paulS.jpg", cv2.IMREAD_COLOR)#
    img = cv2.imread("rowS.jpg", cv2.IMREAD_COLOR)#

    #  img = cv2.imread("deskS.jpg", cv2.IMREAD_COLOR)
    # img = cv2.imread("sudoku.png", cv2.IMREAD_COLOR)

    # warpPoints = np.array([(73, 239), (356, 117), (475, 265), (187, 443)])#(187, 443)])
    # warpPoints = np.array([(78, 279), (378, 148), (504, 298), (199, 489)])  # perTest2
    # warpPoints = np.array([(2148, 1716), (3372, 2092), (2952, 2588), (1600, 2020) ])  # iPad
    # warpPoints = np.array([(538, 430), (840, 522), (743, 645), (400, 510)])  # iPadSmall
    # warpPoints = np.array([(430, 538), (522, 840), (645, 743), (510, 400)])  # iPadSmall Y,x
    # warpPoints = np.array([(112, 140), (447, 140), (516, 381), (116, 396)])  # sudoko
    # warpPoints = np.array([(219, 0), (826, 0), (800, 236), (252, 246)])  # iPadSmall (Monitor)
    # warpPoints = np.array([(538, 430), (840, 478), (800, 616), (432, 560)])  # iPadSmall
    # warpPoints = np.array([(732, 918), (990, 956), (884, 1165), (504, 1118)])  # desksmall
    # warpPoints = np.array([(323, 265), (608, 273), (771, 600), (47, 596)])  # paulS
    warpPoints = np.array([(436, 589), (564, 590), (577, 654), (424, 655)])  # paulS

    warped_full, warped_part = four_point_transform(img, warpPoints)

    for (x, y) in warpPoints:
        cv2.circle(img, (x, y), 5, (0, 255, 0), 10)

    # cv2.imshow('Original', img)

    # warped_full = autocrop(warped_full)#get_cropped_img(warped_full)

    r = 2000 / warped_full.shape[1]
    dim = (2000, int(warped_full.shape[0] * r))
    smallWarped = cv2.resize(warped_full, dim, interpolation=cv2.INTER_AREA)
    # cv2.imshow('warped_full', warped_full)
    # cv2.imshow('Warped_full', smallWarped)
    cv2.imwrite('Warped.jpg', warped_full)
    #
    # r = 600.0 / warped_part.shape[1]
    # dim = (600, int(warped_part.shape[0] * r))
    # smallWarped_p = cv2.resize(warped_part, dim, interpolation=cv2.INTER_AREA)
    # cv2.imshow('warped_part', warped_part)
    # cv2.imshow('Warped_part', smallWarped_p)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
