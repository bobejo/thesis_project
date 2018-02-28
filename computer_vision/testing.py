import numpy as np
import cv2
import cnn
import paths
import img_numpy
import grasp_finder as gf
from matplotlib import pyplot as plt
#from grasp_finder import featurematching_coordinates, least_square_solver, affine_transformation, triangulate_point, contour_detector
#from grasp_finder import binary_image, dilate_image, image_segmentation, blob_detector, find_contact_points, create_square
from keras.models import load_model
from Loss import LogLoss, accuracy


def test_triangulation():
    """
    Triangulates using the affine transformation.
    Print the global coordinates

    :return:
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
    x_test_path=paths.x_test_path+'\\inp\\*.jpg'
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


def test_affine():
    """
    Finds similarities between the two images and uses this to find the affine transformation between the images.
    Plots the true coordinate in the left image and the transformed in the right image.

    :return:
    """
    # The path for images taken with both cameras at the same time
    test_path_right = paths.test_path_right
    test_path_left = paths.test_path_left

    [lpt, rpt] = gf.featurematching_coordinates(test_path_left, test_path_right, 31)
    A,t=gf.affine_transformation_solver(lpt, rpt)
    #A, t = gf.least_square_solver(lpt, rpt)

    for i in range(0, 15):
        img1 = cv2.imread(test_path_left, 0)
        img2 = cv2.imread(test_path_right, 0)

        left_points = lpt[i]
        right_points = gf.affine_transformation(A, t, left_points)
        print('==========================')
        print('True left ' + str(lpt[i]))
        print('True right' + str(rpt[i]))
        print('Estimated right points ' + str(right_points))
        print('==========================')

        left_points = int(lpt[i][0]), int(lpt[i][1])
        right_points = int(rpt[i][0]), int(rpt[i][1])
        cv2.circle(img1, left_points, 3, (255, 0, 0), 5)
        cv2.circle(img2, right_points, 3, (0, 0, 0), 5)
        fig1, axs1 = plt.subplots(1, 2, figsize=(20, 20))

        axs1[0].imshow(img1, cmap='gray')
        axs1[1].imshow(img2, cmap='gray')
        plt.tight_layout()
        plt.show()


def test_contour():
    """
    Contour test


    :return:
    """

    model = load_model(paths.model_path, custom_objects={'LogRegLoss': LogLoss()})
    x_test_path = paths.x_test_path
    y_test_path = paths.y_test_path

    test_generator = cnn.create_generators(x_test_path, y_test_path, 1)
    for i in range(0, 100):
        (inp, target) = next(test_generator)

        p = cnn.get_prediction(model, inp)
        bi = gf.binary_image(p[0], 0.2)
        di = gf.dilate_image(bi, 5)
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

    :return:
    """

    model = load_model(paths.model_path, custom_objects={'LogRegLoss': LogLoss()})
    x_test_path = paths.x_test_path
    y_test_path = paths.y_test_path

    test_generator = cnn.create_generators(x_test_path, y_test_path, 1)
    for i in range(0, 100):
        (inp, target) = next(test_generator)

        p = cnn.get_prediction(model, inp)
        bi = gf.binary_image(p[0], 0.2)
        di = gf.dilate_image(bi, 5)
        k = gf.blob_detector(di)
        im_with_keypoints = cv2.drawKeypoints(di, k, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

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

    :return:
    """
    x_test_path = paths.x_test_path
    y_test_path = paths.y_test_path
    test_generator = cnn.create_generators(x_test_path, y_test_path, 1)


    for i in range(0,10):
        fig, axs = plt.subplots(1, 2, figsize=(20, 20))
        (inp, target) = next(test_generator)
        axs[0].imshow(inp[0])
        axs[1].imshow(target[0].reshape(550,400), cmap='gray')
        plt.tight_layout()
        plt.show()


def test_contact_points():
    """
    Uses the found blobs to find the two contact points of the image.
    Plots the dilated image together with the contact points

    :return:
    """

    model = load_model(paths.model_path, custom_objects={'LogRegLoss': LogLoss()})
    x_test_path = paths.x_test_path
    y_test_path = paths.y_test_path

    test_generator = cnn.create_generators(x_test_path, y_test_path, 1)
    for i in range(0,10):
        (inp, target) = next(test_generator)
        p = cnn.get_prediction(model, inp)
        bi = gf.binary_image(p[0], 0.2)
        di = gf.dilate_image(bi, 5)
        k = gf.blob_detector(di)

        pt = (int(k[-1].pt[0]), int(k[-1].pt[1]))
        contacts = gf.find_contact_points(di, pt)
        cv2.circle(di, contacts[0], 2, 100, 3)
        cv2.circle(di, contacts[1], 2, 100, 3)
        cv2.imshow('Contact points', di)
        cv2.waitKey(0)



#test_generation()
#test_affine()
#test_contour()
#test_contact_points()
#test_blobdetection()
test_prediction()