import numpy as np
import cv2
import cnn
import paths
import img_numpy
import transformation as tf
import grasp_finder as gf
import image_segmentation as iseg
from matplotlib import pyplot as plt
from keras.models import load_model
from Loss import LogLoss, accuracy

'''
Contains test for several functions
'''

batch_size = 1
row_size = 500
col_size = 400


def test_triangulation():
    """
    Triangulates using the affine transformation.
    Print the global coordinates
    """

    # The path for images taken with both cameras at the same time
    test_path_right = paths.test_path_right
    test_path_left = paths.test_path_left

    for i in range(0, 15):
        [lpt, rpt] = gf.featurematching_coordinates(test_path_left, test_path_right, 30)
        lcm = np.load(paths.left_matrix_path)
        rcm = np.load(paths.right_matrix_path)
        ltri = np.add(lpt[i], (400, 930))
        rtri = np.add(rpt[i], (400, 1380))
        tri = gf.triangulate_point(ltri, rtri, lcm, rcm)
        print('X ' + str(tri[0][0]))
        print('Y ' + str(tri[1][0]))
        print('Z ' + str(tri[2][0]))


def test_prediction():
    """
    Plots the input, target and prediction images.
    Runs get_prediction() to get the prediction.

    Press ANY button to change images.
    ESC to exit
    """
    x_test_path = paths.x_test_path + '\\inp\\*.jpg'
    y_test_path = paths.y_test_path + '\\inp\\*.jpg'
    model = load_model(paths.model_path, custom_objects={'LogRegLoss': LogLoss()})
    x_test = img_numpy.imgs2numpy(x_test_path, 100)
    y_test = img_numpy.imgs2numpy(y_test_path, 100)
    p = cnn.get_prediction(model, x_test)
    for i in range(0, 100):
        cv2.imshow('input', x_test[i])
        cv2.imshow('target', y_test[i])
        cv2.imshow('output', p[i])
        cv2.waitKey(0)


def test_transformation():
    """
    Finds similarities between the two images. Uses RANSAC to remove outliers. Least square solver is used for the
    inliers to estimate the transformation.
    Plots the true coordinate in the left image and the transformed in the right image.
    """
    # The path for images taken with both cameras at the same time
    test_path_right = paths.test_path_right
    test_path_left = paths.test_path_left

    [lpt, rpt] = gf.featurematching_coordinates(test_path_left, test_path_right, 30)
    A, t = tf.least_square_solver(lpt, rpt,20)

    for i in range(0, len(lpt)):
        img1 = cv2.imread(test_path_left, 0)
        img2 = cv2.imread(test_path_right, 0)

        left_points = lpt[i]
        right_points = tf.affine_transformation(A, t, left_points)
        print('==========================')
        print('True left points ' + str(lpt[i]))
        print('True right points ' + str(rpt[i]))
        print('Estimated right points ' + str(right_points))
        print('==========================')

        left_points = int(lpt[i][0]), int(lpt[i][1])
        right_points = int(rpt[i][0]), int(rpt[i][1])
        cv2.circle(img1, left_points, 3, (255, 0, 0), 5)
        cv2.circle(img2, right_points, 3, (0, 0, 0), 5)
        cv2.imshow('left',img1)
        cv2.imshow('right',img2)
        cv2.waitKey(0)


def test_contour():
    """
    Contour test
    """

    model = load_model(paths.model_path, custom_objects={'LogRegLoss': LogLoss()})
    x_test_path = paths.x_test_path
    y_test_path = paths.y_test_path

    test_generator = cnn.create_generators(x_test_path, y_test_path, 1)
    for i in range(0, 100):
        (inp, target) = next(test_generator)

        p = cnn.get_prediction(model, inp)
        bi = iseg.binary_image(p[0], 0.2)
        di = iseg.dilate_image(bi, 5)
        cont = gf.contour_detector(di)

        fig3, axs3 = plt.subplots(2, 2, figsize=(30, 30))

        # Binary image and dilation
        axs3[0][0].imshow(bi, cmap='gray')
        axs3[0][1].imshow(di, cmap='gray')
        axs3[1][0].imshow(inp[0])

        plt.show()


def test_blobdetection():
    """
    Test the blob detection for several images

    """

    model = load_model(paths.model_path, custom_objects={'LogRegLoss': LogLoss()})
    x_test_path = paths.x_test_path
    y_test_path = paths.y_test_path

    test_generator = cnn.create_generators(x_test_path, y_test_path, 1)
    for i in range(0, 100):
        (inp, target) = next(test_generator)

        p = cnn.get_prediction(model, inp)
        bi = iseg.binary_image(p[0], 0.2)
        di = iseg.dilate_image(bi, 5)
        k = gf.blob_detector(di)
        im_with_keypoints = cv2.drawKeypoints(di, k, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        fig3, axs3 = plt.subplots(2, 2, figsize=(30, 30))

        # Binary image and dilation
        axs3[0][0].imshow(bi, cmap='gray')
        axs3[0][1].imshow(di, cmap='gray')

        # Blob detection and input image
        axs3[1][0].imshow(inp[0])
        axs3[1][1].imshow(im_with_keypoints)

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
test_transformation()
# test_contour()
# test_contact_points()
# test_blobdetection()
# test_prediction()
