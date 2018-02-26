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
    di = cv2.dilate(img, kernel, iterations=1)
    dilated_img = di.astype(np.uint8)
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
    # params.filterByArea = True
    # params.minArea = 100
    # params.maxArea = 5000
    params.minDistBetweenBlobs = 1

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    keypoints.sort(key=operator.attrgetter('size'))

    k = keypoints
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("Keypoints", im_with_keypoints)
    return k


def affine_transformation(A, t, lpoints):
    """
    Calculates the coordinates of the right image using affine transformation
    rpoints=A*lpoints+t

    :param A: Transformation matrix (2x2)
    :param t: Translation vector
    :param lpoints:
    :return:
    """

    mul = np.matmul(A.reshape(2, 2), lpoints)
    rpoints = np.add(mul.reshape(2, 1), t)
    return (rpoints[0], rpoints[1])


def least_square_solver(lpoints, rpoints):
    """
    Uses the least square method the estimate the transformation matrix and translation vector

    :param lpoints: List of coordinates (x,y), of at least length 3, for left camera
    :param rpoints: List of coordinates (x,y), of at least length 3, for right camera
    :return: Transformation matrix A and translation vector t
    """
    N = len(lpoints)
    M = np.array([[lpoints[0][0], lpoints[0][1], 1, 0, 0, 0], [0, 0, 0, lpoints[0][0], lpoints[0][1], 1]])
    v = np.array([[rpoints[0][0]], [rpoints[0][1]]])

    for i in range(1, N):
        M_row = np.array([[lpoints[i][0], lpoints[i][1], 1, 0, 0, 0], [0, 0, 0, lpoints[i][0], lpoints[i][1], 1]])
        v_row = np.array([[rpoints[i][0]], [rpoints[i][1]]])
        M = np.vstack((M, M_row))
        v = np.vstack((v, v_row))

    theta, r, ra, s = np.linalg.lstsq(M, v)
    print(theta)
    A = np.array([[theta[0], theta[1]], [theta[3], theta[4]]])
    t = np.array([theta[2], theta[5]])
    return A, t


def affine_transformation_solver(lpoints, rpoints):
    """
    Calculates the transformation matrix and translation vector with 3 coordinates.
    theta=M\v

    :param lpoints:  List of coordinates (x,y), of at least length 3, for left camera
    :param rpoints: List of coordinates (x,y), of at least length 3, for right camera
    :return: Transformation matrix A and translation vector t
    """
    M = np.matrix([[lpoints[0][0], lpoints[0][1], 0, 0, 1, 0],
                   [0, 0, lpoints[0][0], lpoints[0][1], 0, 1],
                   [lpoints[1][0], lpoints[1][1], 0, 0, 1, 0],
                   [0, 0, lpoints[1][0], lpoints[1][1], 0, 1],
                   [lpoints[2][0], lpoints[2][1], 0, 0, 1, 0],
                   [0, 0, lpoints[2][0], lpoints[2][1], 0, 1]]
                  )

    v = np.matrix(
        [[rpoints[0][0]], [rpoints[0][1]], [rpoints[1][0]], [rpoints[1][1]], [rpoints[7][0]],
         [rpoints[7][1]]])
    theta = solve(M, v)
    A = np.array([theta[0], theta[1], theta[2], theta[3]])
    A = A.reshape(2, 2)
    t = np.array([theta[4], theta[5]])
    return A, t


def featurematching_coordinates(limg, rimg, threshold=10):
    """
    Extracts features from each image prediction and matches them with eachother to find similarities.
    Sorts the matches and returns all coordinates off all matches

    :param limg:   The image from the left camera
    :param rimg:   The image from the right camera
    :param threshold: The maximum distance between descriptors
    :return: (left_coordinates, right_coordinates): The coordinates for the each match for the images
    """

    img1 = cv2.imread(limg, 0)
    img2 = cv2.imread(rimg, 0)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good = []

    for m in matches:
        if m.distance < threshold:
            good.append([m])

    good = sorted(good, key=lambda x: x[0].distance)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    # plt.imshow(img3), plt.show()

    left_coord = [kp1[mat[0].queryIdx].pt for mat in good]
    right_coord = [kp2[mat[0].trainIdx].pt for mat in good]

    return left_coord, right_coord


def find_contact_points(img, center):
    """
    Finds the two contact points for the gripper

    :param img: The binary dilated image as numpy
    :param center: The center of the previously extracted blob
    :return: The two contacts points
    """
    print(center)
    remove = 0
    contacts = []
    removed = []
    coord = (0, 0)
    combinations=create_square(i)

    prev_combinations = [(0, 0)]
    for i in range(1, 10):
        if len(contacts) > 1:
            print('Contact points found')
            return contacts
        if remove:
            ind=combinations.index(remove)
            print(removed)

        combinations = create_square(i)

        if ind:


        for w, h in combinations:
            coord = (center[0] + w, center[1] + h)

            if img[coord[1]][coord[0]] == 0:
                print('Found, removing next loop')
                contacts.append((coord[0], coord[1]))
                # print(contacts)
                remove = (w,h)
                break

    return False
def create_square(size):
    top=[]
    right = []
    bottom = []
    left = []
    aranged=np.arange(-size, size + 1)
    for i in aranged:
        top=top+[(-size,i)]
        bottom=bottom+[(size,i)]
        right=right+[(i,size)]
        left = left + [(i, -size)]
    bottom.reverse()
    left.reverse()
    square=top+right[1:]+bottom[1:]+left[1:-1]

    print(square)
    print('=================')
    print('top'+str(top))
    print('right' + str(right))
    print('bottom' + str(bottom))
    print('left' + str(left))
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
