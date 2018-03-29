# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import cv2

import cnn
import paths
import img_numpy
import image_registration as tf
import grasp_finder as gf
import image_manipulation as iseg
import global_variables as gv
from matplotlib import pyplot as plt
from keras.models import load_model
from Loss import LogLoss, accuracy
from mpl_toolkits.mplot3d import Axes3D

'''
Contains test for several functions
'''
batch_size = 1
row_size = 550
col_size = 400


def test_cropping():
    """
    Plots the points in the cropped image and the full image
    :return:
    """
    pathleft = 'C:\\Users\\Samuel\\Desktop\\pipes\\left\\images\\*.jpg'
    pathright = 'C:\\Users\\Samuel\\Desktop\\pipes\\right\\images\\*.jpg'
    # ci.crop_images(pathleft)
    # ci.crop_images(pathright)

    [lp, rp] = gf.featurematching_coordinates(paths.test_path_left1, paths.test_path_right1, 40)
    print(len(lp))
    A, t = tf.least_square_solver(lp, rp, 550)
    croppedleft = cv2.imread(paths.test_path_left1, 0)
    fullleft = cv2.imread(paths.test_path_left2, 0)
    croppedright = cv2.imread(paths.test_path_right1, 0)
    fullright = cv2.imread(paths.test_path_right2, 0)

    for i in range(0, len(lp)):
        left_points = lp[i]
        right_points = tf.affine_transformation(A, t, left_points)
        left_points = int(left_points[0]), int(left_points[1])
        right_points = int(right_points[0]), int(right_points[1])
        cv2.circle(croppedleft, left_points, 2, (255, 0, 0), 3)
        cv2.circle(croppedright, right_points, 2, (255, 0, 0), 3)

        ltri = np.add(left_points, (1380, 400))
        rtri = np.add(right_points, (930, 400))
        cv2.circle(fullleft, tuple(ltri), 2, (255, 0, 0), 3)
        cv2.circle(fullright, tuple(rtri), 2, (255, 0, 0), 3)

    fig, axs = plt.subplots(2, 2, figsize=(30, 30))
    axs[0][0].imshow(croppedleft, cmap='gray')
    axs[0][1].imshow(fullleft, cmap='gray')
    axs[1][0].imshow(croppedright, cmap='gray')
    axs[1][1].imshow(fullright, cmap='gray')

    plt.show()


def test_triangulation():
    """
    Triangulates all the points from left and right image. Plots the image points and the 3d points in robot base frame and left camera frame.

    """

    # Create figures
    fig = plt.figure('Camera frame')
    fig1 = plt.figure('Base frame')
    fig_right = plt.figure('Right camera')
    fig_left = plt.figure('Left camera')
    ax = fig.add_subplot(111, projection='3d')
    ax1 = fig1.add_subplot(111, projection='3d')
    ax_right = fig_right.add_subplot(111)
    ax_left = fig_left.add_subplot(111)


    # Load camera matrices and points
    lcm = np.genfromtxt('lcm_vlh2.txt')
    rcm = np.genfromtxt('rcm_vlh2.txt')

    # When using function  choose_points
    #lpt = [tuple(lp) for lp in np.load('C:\\Users\\Samuel\\Desktop\\pipes\\left\\images\\left2.npy').tolist()]
    #rpt = [tuple(lp) for lp in np.load('C:\\Users\\Samuel\\Desktop\\pipes\\right\\images\\right2.npy').tolist()]

    lpt = np.genfromtxt('lpoints3.txt')

    # Choose if right points should be true points or transformed
    rpt = np.genfromtxt('rpoints3.txt')
    # rpt = [tf.affine_transformation(gv.A, gv.t, l) for l in lpt]

    # Load and undistort the images
    img_left = cv2.imread(paths.left_chessboard2)
    img_right = cv2.imread(paths.right_chessboard2)
    undistimg_left = cv2.undistort(img_left, gv.K1, gv.d1, None)
    undistimg_right = cv2.undistort(img_right, gv.K2, gv.d2, None)

    for i in range(0, len(lpt)):
        # Triangulate and transform to baseframe
        tri = gf.triangulate_point(lpt[i], rpt[i], rcm, lcm)
        tri_base = tf.base_transform(gv.T, tri)

        ax_left.scatter(lpt[i][0], lpt[i][1], linewidths=10)
        ax_right.scatter(rpt[i][0], rpt[i][1], linewidths=10)
        ax.scatter(tri[0], tri[1], tri[2], marker=',', linewidths=15)
        ax1.scatter(tri_base[0], tri_base[1], tri_base[2], marker=',', linewidths=15)

    # Plot the cameras and baseframe
    """
    camera1 = np.linalg.lstsq(lcm[:3, :3], lcm[:, 3])
    camera2 = np.linalg.lstsq(rcm[:3, :3], rcm[:, 3])
    camera1base = tf.base_transform(T, np.vstack(camera1[0]))
    camera2base = tf.base_transform(T, np.vstack(camera2[0]))
    ax.scatter(camera1[0][0], camera1[0][1], camera1[0][2], c='k', marker='H', linewidths=20)
    ax.scatter(camera2[0][0], camera2[0][1], camera2[0][2], c='k', marker='H', linewidths=20)
    ax1.scatter(0, 0, 0, c='k', marker='H', linewidths=20)
    ax1.scatter(camera1base[0]+250, camera1base[1], camera1base[2], c='k', marker='H', linewidths=20)
    ax1.scatter(camera2base[0]+250, camera2base[1], camera2base[2], c='k', marker='H', linewidths=20)
    """

    ax_left.imshow(undistimg_left)
    ax_right.imshow(undistimg_right)
    ax_right.set_xlabel('X Label')
    ax_right.set_ylabel('Y Label')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    ax1.set_zlabel('Z Label')
    ax.set_title('Positions in camera frame')
    ax1.set_title('Positions in robot base frame')
    fig.gca().invert_xaxis()
    plt.show()


def test_prediction():
    """
    Plots the input, target and prediction images.
    Runs get_prediction() to get the prediction.

    Press ANY button to change images.
    ESC to exit
    """
    x_test_path = paths.x_test_path
    y_test_path = paths.y_test_path
    test_generator = cnn.create_generators(x_test_path, y_test_path, batch_size, row_size, col_size)
    model = load_model(paths.model_path, custom_objects={'LogRegLoss': LogLoss()})

    for i in range(0, 100):
        x, target = next(test_generator)
        p = cnn.get_prediction(model, x)
        bin = iseg.binary_image(p[0], 0.25)
        cv2.imshow('input', x[0])
        cv2.imshow('target', target[0])
        cv2.imshow('output', p[0])
        cv2.imshow('Binary', bin)
        cv2.waitKey(0)


def test_transformation():
    """
    Finds similarities between the two images. Uses RANSAC to remove outliers. Least square solver is used for the
    inliers to estimate the transformation.
    Plots the true coordinate in the left image and the transformed in the right image.
    """
    # The path for images taken with both cameras at the same time
    test_path_right = paths.save_path + '\\right\\rightcalibration09_38_49.jpg'
    test_path_left = paths.save_path + '\\left\\leftcalibration09_38_49.jpg'
    lpt = np.genfromtxt('lpoints.txt')
    rpt = np.genfromtxt('rpoints.txt')
    print(len(lpt))
    A, t = tf.least_square_solver(lpt, rpt, 100)

    for i in range(0, len(lpt)):
        img1 = cv2.imread(test_path_left, 0)
        img2 = cv2.imread(test_path_right, 0)

        left_points = lpt[i]
        right_points = tf.affine_transformation(A, t, left_points)

        print('==========================')
        print('True left points ' + str(left_points))
        print('True right points ' + str(rpt[i]))
        print('Estimated right points ' + str(right_points))
        print('==========================')

        left_points = int(left_points[0]), int(left_points[1])
        right_points = int(right_points[0]), int(right_points[1])
        cv2.circle(img1, left_points, 3, (0, 0, 0), 5)
        cv2.circle(img2, right_points, 3, (0, 0, 0), 5)

        cv2.imshow('left', img1)
        cv2.imshow('right', img2)
        cv2.waitKey(0)


def test_contour():
    """
    Contour test
    """
    model = load_model(paths.model_path, custom_objects={'LogRegLoss': LogLoss()})
    x_test_path = paths.x_train_path
    y_test_path = paths.y_train_path

    test_generator = cnn.create_generators(x_test_path, y_test_path, 1, row_size, col_size)
    for i in range(0, 100):
        (inp, target) = next(test_generator)

        p = cnn.get_prediction(model, inp)
        bi = iseg.binary_image(p[0], 0.10)
        di = iseg.dilate_image(bi, 5)
        img, mom, ang = gf.contour_detector(di)
        print('Angle: ', ang)
        print('Point', mom)
        cv2.imshow('cont', img)
        cv2.circle(inp[0], mom, 3, (255, 0, 0), 3)
        cv2.imshow('inp', inp[0])
        cv2.waitKey(0)


def test_blobdetection():
    """
    Test the blob detection for several images

    """

    model = load_model(paths.model_path, custom_objects={'LogRegLoss': LogLoss()})
    x_test_path = paths.x_test_path
    y_test_path = paths.y_test_path

    test_generator = cnn.create_generators(x_test_path, y_test_path, 1, 550, 400)
    for i in range(0, 100):
        (inp, target) = next(test_generator)

        p = cnn.get_prediction(model, inp)
        bi = iseg.binary_image(p[0], 0.2)
        di = iseg.dilate_image(bi, 5)
        k = gf.blob_detector(di)
        im_with_keypoints = cv2.drawKeypoints(di, k, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('blobs', im_with_keypoints)
        cv2.waitKey(0)
        # fig3, axs3 = plt.subplots(2, 2, figsize=(30, 30))
        plt.imshow(di, cmap='gray')
        # Binary image and dilation
        # axs3[0][0].imshow(bi, cmap='gray')
        # axs3[0][1].imshow(di, cmap='gray')

        # Blob detection and input image
        # axs3[1][0].imshow(inp[0])
        # axs3[1][1].imshow(im_with_keypoints)

        plt.show()


def test_generation():
    """
    Creates a generator and plots the output.
    """

    x_test_path = paths.x_test_path
    y_test_path = paths.y_test_path
    test_generator = cnn.create_generators(x_test_path, y_test_path, batch_size, row_size, col_size)

    for i in range(0, 10):
        fig, axs = plt.subplots(1, 2, figsize=(20, 20))
        (inp, target) = next(test_generator)
        cv2.imshow('inp', inp[0])
        cv2.imshow('target', target[0])
        cv2.waitKey(0)
        axs[0].imshow(inp[0])
        axs[1].imshow(target[0].reshape(550, 400), cmap='gray')
        plt.tight_layout()
        plt.show()


def test_contact_points():
    """
    Uses the found blobs to find the two contact points of the image.
    Plots the dilated image together with the contact points
    """

    model = load_model(paths.model_path, custom_objects={'LogRegLoss': LogLoss()})
    x_test_path = paths.x_test_path
    y_test_path = paths.y_test_path

    test_generator = cnn.create_generators(x_test_path, y_test_path, 1)
    for i in range(0, 25):
        (inp, target) = next(test_generator)
        p = cnn.get_prediction(model, inp)
        bi = iseg.binary_image(p[0], 0.2)
        di = iseg.dilate_image(bi, 5)
        k = gf.blob_detector(di)

        pt = (int(k[-1].pt[0]), int(k[-1].pt[1]))
        contacts = gf.find_contact_points(di, pt)
        cv2.circle(di, contacts[0], 2, 100, 3)
        cv2.circle(di, contacts[1], 2, 100, 3)
        cv2.imshow('Contact points', di)
        cv2.waitKey(0)


# test_generation()
# test_transformation()
# test_contour()
# test_contact_points()
# test_blobdetection()
# test_prediction()

test_triangulation()
# test_cropping()
# lpt = np.genfromtxt('lpoints3.txt')
# tf.projective_transformation(gv.H,lpt[0])
