import numpy as np
from scipy.linalg import solve
import random


def camera_transform(T, U_camera):
    """

    :param T: Transformation matrix (3x4)
    :param U: Points (X;Y;Z)
    :return:
    """

    U_camera = np.vstack([U_camera, 1])
    T = np.vstack([T, [0, 0, 0, 1]])
    return np.dot(T, U_camera)

def affine_transformation(A, t, lpoints):
    """
    Calculates the coordinates of the right image using affine transformation
    rpoints=A*lpoints+t

    :param A: Transformation matrix (2x2)
    :param t: Translation vector
    :param lpoints: Points from left image
    :return: The corresponding points in the right image
    """
    mul = np.matmul(A.reshape(2, 2), lpoints)
    rpoints = np.add(mul.reshape(2, 1), t)
    return tuple((rpoints[0][0], rpoints[1][0]))


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
        [[rpoints[0][0]], [rpoints[0][1]], [rpoints[1][0]], [rpoints[1][1]], [rpoints[2][0]],
         [rpoints[2][1]]])
    try:
        theta = solve(M, v)
    except np.linalg.linalg.LinAlgError:
        return np.array([0, 0, 0, 0]), np.array([0, 0])

    A = np.array([theta[0], theta[1], theta[2], theta[3]])
    A = A.reshape(2, 2)
    t = np.array([theta[4], theta[5]])
    return A, t


def get_residuals(A, t, lpoints, rpoints):
    """
    Estimates the residuals for the transformation

    :param A: Transformation matrix
    :param t: Translation vector
    :param lpoints: Points from left image that were used for calculating A and t
    :param rpoints: Points from right image
    :return: The absolute residual
    """

    ri = np.empty((2, len(lpoints)))
    for i in range(0, len(lpoints)):
        affine = affine_transformation(A, t, lpoints[i])
        ri[0][i] = affine[0] - rpoints[i][0]
        ri[1][i] = affine[1] - rpoints[i][1]
    return np.sqrt(np.sum(np.multiply(ri, ri)))


def ransac_inliers(lpoints, rpoints, threshold):
    """
    Takes matched points from feature_match, does affine transformation and removes points which gives outliers.

    :param lpoints: Left image features from feature_matching
    :param rpoints: Right image features from feature_matching
    :param threshold: The threshold for classifying points as outliers
    :return: The inliers for left and right points
    """

    N = len(lpoints)
    new_lpoints = []
    new_rpoints = []
    loop_length = 10000 + 500 * N
    all_loss = []

    for i in range(0, loop_length):
        lsample = []
        rsample = []
        index = []
        j = 0
        while j < 3:
            rand_index = random.randrange(0, len(lpoints) - 3)
            if rand_index not in index:
                lsample.append(lpoints[rand_index])
                rsample.append(rpoints[rand_index])
                index.append(rand_index)
                j += 1
        [A, t] = affine_transformation_solver(lsample, rsample)

        loss = get_residuals(A, t, lpoints, rpoints)
        all_loss = all_loss + [loss]

        if loss < threshold:
            new_lpoints.append(lsample)
            new_rpoints.append(rsample)

    unique_lpoints = []
    for tup in new_lpoints:
        for i in range(0, 3):
            if tuple(tup[i]) not in unique_lpoints:
                unique_lpoints.append(tuple(tup[i]))

    unique_rpoints = []

    for tup in new_rpoints:
        for i in range(0, 3):
            if tuple(tup[i]) not in unique_rpoints:
                unique_rpoints.append(tuple(tup[i]))

    return unique_lpoints, unique_rpoints


def least_square_solver(lpoints, rpoints, threshold):
    """
    Uses the least square method the estimate the transformation matrix and translation vector.
    First removes outliers using ransac_inliers

    :param lpoints: List of coordinates (x,y), of at least length 3, for left camera
    :param rpoints: List of coordinates (x,y), of at least length 3, for right camera
    :param threshold: The treshold used for removing outliers in RANSAC
    :return: Transformation matrix A and translation vector t
    """
    lpoints, rpoints = ransac_inliers(lpoints, rpoints, threshold)
    print('Ransac points ' + str(len(lpoints)))
    N = len(lpoints)
    if N < 3:
        print('Not enough inliers found!')
        return None, None

    M = np.array([[lpoints[0][0], lpoints[0][1], 1, 0, 0, 0], [0, 0, 0, lpoints[0][0], lpoints[0][1], 1]])
    v = np.array([[rpoints[0][0]], [rpoints[0][1]]])

    for i in range(1, N):
        M_row = np.array([[lpoints[i][0], lpoints[i][1], 1, 0, 0, 0], [0, 0, 0, lpoints[i][0], lpoints[i][1], 1]])
        v_row = np.array([[rpoints[i][0]], [rpoints[i][1]]])
        M = np.vstack((M, M_row))
        v = np.vstack((v, v_row))

    theta, r, ra, s = np.linalg.lstsq(M, v)
    A = np.array([[theta[0], theta[1]], [theta[3], theta[4]]])
    t = np.array([theta[2], theta[5]])
    return A, t
