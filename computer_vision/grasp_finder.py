# Standard imports
import cv2
import numpy as np;
from keras.models import load_model
import cnn
from img_numpy import imgs2numpy
from Loss import LogLoss, accuracy
from matplotlib import pyplot as plt
from scipy import ndimage
import operator
from scipy.linalg import solve


def binary_image(img, threshold):
    """
    Set all parameters below the threshold to 0 and all above to 1

    :param img: The image
    :param threshold: The threshold
    :return: An binary image where each pixel is either 1 or 0
    """
    r, bi = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return bi


def dilate_image(img, size):
    """
    Dilates the binary image, making the lines thicker and more connected

    :param img: Binary image
    :param size: Size of the dilation
    :return: Dilated binary image
    """
    kernel = np.ones((size, size))
    dilated_img = cv2.dilate(img, kernel, iterations=1)
    return dilated_img


def image_segmentation(img, threshold, size):
    """
    Does image segmentation on the input image. First converts to binary image then dilates it
    :param img: The output from prediction
    :param threshold: The threshold for choosing 1 and 0
    :param size: The size of the dilation
    :return: A binary dilated image of type uint8 (wanted by blobdetector)
    """

    bi = binary_image(img[0], threshold)
    di = dilate_image(bi, size)
    dim = di.astype(np.uint8)
    return dim


def blob_detector(img):
    """
    Plots the blobs of the image

    :param img: Binary dilated image
    :return:
    """
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
    params.minArea = 400
    params.maxArea = 3500
    params.minDistBetweenBlobs = 10
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    keypoints.sort(key=operator.attrgetter('size'))
    if len(keypoints) != 0:
        keypoints = [keypoints[-1]]
        k = keypoints[-1]
    else:
        keypoints = []
        k = 0
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints", im_with_keypoints)
    return k


def affine_transformation(points, tpoints):
    """
    Calculates the transformation matrix and translation vector with 3 coordinates
    tpoints=A*points+t

    :param points: List of coordinates (x,y), minimum length 3
    :param tpoints: List of transformed coordinates (x,y) minimum length 3
    :return: Transformation matrix A and translation vector t
    """

    M = np.matrix([[points[0][0], points[0][1], 0, 0, 1, 0],
                   [0, 0, points[0][0], points[0][1], 0, 1],
                   [points[1][0], points[1][1], 0, 0, 1, 0],
                   [0, 0, points[1][0], points[1][1], 0, 1],
                   [points[2][0], points[2][1], 0, 0, 1, 0],
                   [0, 0, points[2][0], points[2][1], 0, 1]]
                  )
    v = np.matrix(
        [[tpoints[0][0]], [tpoints[0][1]], [tpoints[1][0]], [tpoints[1][1]], [tpoints[2][0]], [tpoints[2][1]]])
    theta = solve(M, v)
    A = np.array([theta[0], theta[1], theta[2], theta[3]])
    A = A.reshape(2, 2)
    t = np.array([theta[4], theta[5]])
    return (A, t)


"""
model = load_model('C:\\Users\\Samuel\\GoogleDrive\\Master\\complete\\CNN\\Trained_Models\Pipe_symmetric.h5',
                   custom_objects={'LogRegLoss': LogLoss()})
testgen = cnn.create_generators(cnn.x_test_path, cnn.y_test_path, 1)

for i in range(0, 50):
    (inp, targ) = next(testgen)
    p = cnn.get_prediction(model, inp)
    bi = binary_image(p[0], 0.17)
    cv2.imshow('bi', bi)
    di = dilate_image(bi, 6)
    di = di.astype(np.uint8)
    print(bi)
    cv2.imshow('Input', inp[0])
    cv2.imshow('target', targ[0])
    cv2.imshow('p', p[0])

    # cv2.imshow('di',di)
    blob_detector(di)

    cv2.waitKey(0)
"""
