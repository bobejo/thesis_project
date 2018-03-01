import cv2
import numpy as np;
from keras.models import load_model
import cnn
from img_numpy import imgs2numpy
from Loss import LogLoss, accuracy
from matplotlib import pyplot as plt
from scipy import ndimage
import operator


def contour_detector(img):
    """
    Finds the contour of the dilated image.

    :param img: Dilated image
    :return: The contour of the image
    """
    # TODO. Fix settings

    cv2.imshow('img', img)
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im2, contours, -1, (0, 255, 0), 3)
    cv2.imshow('im2', im2)
    # cv2.imshow('cont',contours)
    cv2.waitKey(0)
    return im2


def blob_detector(img):
    """
    Creates a blob detector and uses it to find the blobs of the binary image

    :param img: Binary dilated image
    :return: The keypoints for the blobs
    """
    # TODO Fix settings
    # Settings for blobdetector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity = False
    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minThreshold = 0
    params.maxThreshold = 255
    params.thresholdStep = 10
    params.filterByArea = True
    params.minArea = 200
    # params.maxArea = 5000
    params.minDistBetweenBlobs = 1

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    keypoints.sort(key=operator.attrgetter('size'))

    k = keypoints

    return k


def featurematching_coordinates(limg, rimg, threshold=10):
    """
    Extracts features from each image prediction and matches them with each other to find similarities.
    Sorts the matches and returns all coordinates off all matches

    :param limg:   The image from the left camera
    :param rimg:   The image from the right camera
    :param threshold: The maximum distance between descriptors
    :return: (lpoints, rpoints): The coordinates for each match in left image and right image
    """

    img1 = cv2.imread(limg, 0)
    img2 = cv2.imread(rimg, 0)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

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

    M = np.zeros((6, 5))
    for row in range(0, 3):
        M[row, 2:] = left_cm[row, :3]
        M[row + 3, 2:] = right_cm[row, :3]
    M[:3, 0] = -np.hstack((lpoint, [1]))
    M[3:, 1] = -np.hstack((rpoint, [1]))

    b = np.zeros(6)
    b[:3] = -left_cm[:, 3].reshape(3)
    b[3:] = -right_cm[:, 3].reshape(3)

    theta = np.linalg.lstsq(M, b)
    return theta[0][0:]
