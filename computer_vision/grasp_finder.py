import cv2
import numpy as np;
# from keras.models import load_model
# import cnn
# from img_numpy import imgs2numpy
# from Loss import LogLoss, accuracy
from matplotlib import pyplot as plt
from scipy import ndimage
import operator
import image_manipulation as im

def circle_detector(img):
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.8, 100, 150, 100, 65, 50, 65)
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles

        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circles
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        dist = 0
        best_i=0
        for i in range(0, len(circles)):
            temp_dist = 0
            for j in range(0, len(circles)):
                temp_dist += np.sqrt(
                    (np.square(circles[i][0] - circles[j][0]) + np.square(circles[i][1] - circles[j][1])))
                if temp_dist > dist:
                    dist = temp_dist
                    best_i = i

        cv2.rectangle(output, (circles[best_i][0] - 5, circles[best_i][1] - 5),
                      (circles[best_i][0] + 5, circles[best_i][1] + 5), (255, 0, 255), -1)

    return (int(circles[best_i][0]), int(circles[best_i][1])), output


def contour_detector(img):
    """
    Finds the contours of the image. Returns the binary image with the contour that have the longest length
     and the centroid of this area.
     Center points are given from the image moments of the contour.
    :param img: A binary dilated numpy image
    :return: The input image with the contour, the centroid of this contour and the angle of the contour.
    """

    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    l_area = 0
    length_contour = 0
    # Take the largest contour
    if not contours:
        print('Error No shapes found!')
        return img, (None, None), None, None
    for c in contours:
        length = cv2.arcLength(c, True)
        if length > length_contour:
            length_contour = length
            cont = c

    cv2.drawContours(img, cont, -1, 128, 2)

    # The coordinates of the center of the contour from the moments.
    M = cv2.moments(cont)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # Try to fit an ellipse inside the contour and give the angle of the largest one
    _, _, angle = cv2.fitEllipse(cont)
    print(angle)

    cv2.circle(img, (cx, cy),3,80,3)
    return img, (cx, cy), angle, cont


def triangulate_point(lpoint, rpoint, left_cm, right_cm):
    """
    Uses triangulation to generate the global 3D coordinates using two 2D (x,y) pixel coordinates.
    Finds the unknown X, Y, Z and lamda by solving two pinhole equations: lamda*(x;y;1)=P*(X;Y;Z;1).



    :param lpoint: The pixels of the left image
    :param rpoint: The pixels of the right image
    :param left_cm: The camera matrix for the left camera
    :param right_cm: The camera matrix for the right camera
    :return: 3D point in left camera frame
    """
    print(lpoint)
    print(rpoint)
    theta = cv2.triangulatePoints(left_cm, right_cm, lpoint, rpoint)

    if theta[3] == 0:
        print('Depth is zero')
        return None
    else:
        theta = theta / theta[3]
        theta = theta[:3]
        return theta
