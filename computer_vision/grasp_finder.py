import cv2
import numpy as np;
# from keras.models import load_model
# import cnn
# from img_numpy import imgs2numpy
# from Loss import LogLoss, accuracy
# from matplotlib import pyplot as plt
from scipy import ndimage
import operator


def contour_detector(img):
    """
    Finds the contours of the image. Returns the binary image with the contour that have the largest area
     and the centroid of this area.
    :param img: A binary dilated numpy image
    :return: The input image with the contour, the centroid of this contour and the angle of the contour.
    """
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    l_area = 0
    for c in contours:
        area = cv2.contourArea(c, False)
        if area > l_area:
            l_area = area
            cont = c

    cv2.drawContours(img, cont, -1, 128, 6)
    M = cv2.moments(cont)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    _, _, angle = cv2.fitEllipse(cont)
    return img, (cx, cy), angle


def blob_detector(img):
    """
    Creates a blob detector and uses it to find the blobs of the binary image

    :param img: Binary dilated image
    :return: The keypoints for the blobs
    """
    # TODO Fix settings
    # Settings for blobdetector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.maxCircularity = 0.8
    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minThreshold = 0
    params.maxThreshold = 255
    params.thresholdStep = 10
    params.filterByArea = True
    params.minArea = 20
    # params.maxArea = 5000
    params.minDistBetweenBlobs = 0

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    keypoints.sort(key=operator.attrgetter('size'))

    k = keypoints

    return k


def featurematching_coordinates(limg, rimg, threshold=10):
    """
    Extracts features from left and right image and matches them with each other to find similarities.
    Sorts the matches and return the coordinates off all matches.

    :param limg:   The image from the left camera
    :param rimg:   The image from the right camera
    :param threshold: The maximum distance between descriptors
    :return: (lpoints, rpoints): The coordinates for each match in left image and right image
    """

    if type(limg) == str:
        limg = cv2.imread(limg, 0)
        rimg = cv2.imread(rimg, 0)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(limg, None)
    kp2, des2 = orb.detectAndCompute(rimg, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good = []

    # Only take those with a good match
    for m in matches:
        if m.distance < threshold:
            good.append([m])

    good = sorted(good, key=lambda x: x[0].distance)

    lpoints = [kp1[mat[0].queryIdx].pt for mat in good]
    rpoints = [kp2[mat[0].trainIdx].pt for mat in good]

    return lpoints, rpoints


def find_contact_points(img, center):
    """
    Finds the two contact points for the gripper.
    If one point is found it stops looking at that side.

    :param img: The binary dilated image as numpy
    :param center: The center of the previously extracted blob
    :return: The two contacts points
    """

    contacts = []
    ind = 0
    removed = []

    for i in range(1, 10):
        if len(contacts) > 1:
            return contacts

        combinations = create_square(i)

        if ind:
            new_ind = ind * 2
            removed = np.arange(new_ind - i * 3 - 1, new_ind + i * 3 + 1)

        combinations = [i for j, i in enumerate(combinations) if j not in removed]

        for w, h in combinations:
            coord = (center[0] + w, center[1] + h)

            if img[coord[1]][coord[0]] == 0:
                contacts.append((coord[0], coord[1]))
                ind = combinations.index((w, h))
                break

    return None


def create_square(size):
    """
    Creates a square of coordinates around (0,0) used for find_contact_points

    example of size 1

    [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

    :param size: The length from the center (0,0)
    :return: List of coordinates starting at (-size,-size)
    """
    top = []
    right = []
    bottom = []
    left = []
    aranged = np.arange(-size, size + 1)
    for i in aranged:
        top = top + [(-size, i)]
        bottom = bottom + [(size, i)]
        right = right + [(i, size)]
        left = left + [(i, -size)]
    bottom.reverse()
    left.reverse()
    square = top + right[1:] + bottom[1:] + left[1:-1]

    return square


def triangulate_point(lpoint, rpoint, left_cm, right_cm):
    """
    Uses triangulation to generate the global 3D coordinates using two 2D (x,y) pixel coordinates.

    :param lpoint: The pixels of the left image
    :param rpoint: The pixels of the right image
    :param left_cm: The camera matrix for the left camera
    :param right_cm: The camera matrix for the right camera
    :return: 3D point in global coordinates
    """

    theta = cv2.triangulatePoints(left_cm, right_cm, lpoint, rpoint)

    if theta[3] == 0:
        print('Depth is zero')
        return None
    else:
        theta = theta / theta[3]
        theta = theta[:3]
        return theta
