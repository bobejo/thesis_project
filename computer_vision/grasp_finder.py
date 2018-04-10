import cv2
import numpy as np;
# from keras.models import load_model
# import cnn
# from img_numpy import imgs2numpy
# from Loss import LogLoss, accuracy
from matplotlib import pyplot as plt
from scipy import ndimage
import operator


def contour_detector(img):
    """
    Finds the contours of the image. Returns the binary image with the contour that have the largest area
     and the centroid of this area.
    :param img: A binary dilated numpy image
    :return: The input image with the contour, the centroid of this contour and the angle of the contour.
    """

    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    l_area = 0
    length_contour = 0
    # Take the largest contour
    if not contours:
        print('Error No shapes found!')
        return img
    for c in contours:
        length = cv2.arcLength(c, True)
        if length > length_contour:
            length_contour = length
            cont = c

    cv2.drawContours(img, cont, -1, 128, 10)

    # The coordinates of the center of the contour from the moments.
    M = cv2.moments(cont)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # Try to fit an ellipse inside the contour and give the angle of the largest one
    _, _, angle = cv2.fitEllipse(cont)

    return img, (cx, cy), angle, cont


def blob_detector(img):
    """
    Creates a blob detector and uses it to find the blobs of the binary image

    :param img: Binary dilated image
    :return: The keypoints for the blobs
    """
    #
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

    img3 = cv2.drawMatchesKnn(limg, kp1, rimg, kp2, good, None, flags=2)
    plt.imshow(img3), plt.show()
    lpoints = [kp1[mat[0].queryIdx].pt for mat in good]
    rpoints = [kp2[mat[0].trainIdx].pt for mat in good]

    return lpoints, rpoints


def triangulate_point(lpoint, rpoint, left_cm, right_cm):
    """
    Uses triangulation to generate the global 3D coordinates using two 2D (x,y) pixel coordinates.

    :param lpoint: The pixels of the left image
    :param rpoint: The pixels of the right image
    :param left_cm: The camera matrix for the left camera
    :param right_cm: The camera matrix for the right camera
    :return: 3D point in left camera frame
    """

    theta = cv2.triangulatePoints(left_cm, right_cm, lpoint, rpoint)

    if theta[3] == 0:
        print('Depth is zero')
        return None
    else:
        theta = theta / theta[3]
        theta = theta[:3]
        return theta
