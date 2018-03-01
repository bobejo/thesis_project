import numpy as np
from scipy.linalg import solve


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
    return rpoints[0], rpoints[1]


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
