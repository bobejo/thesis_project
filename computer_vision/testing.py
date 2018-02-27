import numpy as np
import cv2
from matplotlib import pyplot as plt
from grasp_finder import featurematching_coordinates, least_square_solver, affine_transformation, triangulate_point
from grasp_finder import binary_image, dilate_image, image_segmentation, blob_detector, find_contact_points, create_square
from keras.models import load_model
import cnn
from Loss import LogLoss, accuracy


def test_affine():
    test_path_right = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\test\\inp\\right\\right\\rightpipe08_41_37_2.jpg'
    test_path_left = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\test\\inp\\left\\left\\leftpipe08_41_37_2.jpg'

    # test_path_right = 'C:\\Users\\Samuel\\Desktop\\pipes\\right\\images\\right_pipe08_09_55.jpg'
    # test_path_left = 'C:\\Users\\Samuel\\Desktop\\pipes\\left\\images\\left_pipe08_09_55.jpg'

    # test_path_right = 'C:\\Users\\Samuel\\Desktop\\pipes\\right\\images\\croppedright_pipe08_09_55.jpg'
    # test_path_left = 'C:\\Users\\Samuel\\Desktop\\pipes\\left\\images\\croppedleft_pipe08_09_55.jpg'

    [lpt, rpt] = featurematching_coordinates(test_path_left, test_path_right, 31)
    A, t = least_square_solver(lpt, rpt)

    for i in range(0, 15):
        img1 = cv2.imread(test_path_left, 0)
        img2 = cv2.imread(test_path_right, 0)

        lM = np.load('CameraMatrix\\leftMatrix.npy')
        rM = np.load('CameraMatrix\\rightMatrix.npy')
        left_points = lpt[i]
        right_points = affine_transformation(A, t, left_points)
        print('==========================')
        print('True left ' + str(lpt[i]))
        print('True right' + str(rpt[i]))
        print('Right points ' + str(right_points))
        print('==========================')
        print('Triangulation')
        ltri = np.add(lpt[i], (400, 930))
        rtri = np.add(rpt[i], (400, 1380))

        triself = triangulate_point(ltri, rtri, lM, rM)
        output = cv2.triangulatePoints(lM, rM, ltri, rtri)
        tri = np.divide(output, output[3])
        print('My function')
        print(triself)
        print('X ' + str(triself[0]))
        print('Y ' + str(triself[1]))
        print('Z ' + str(triself[2]))
        print('Open cv')
        print('X ' + str(tri[0][0]))
        print('Y ' + str(tri[1][0]))
        print('Z ' + str(tri[2][0]))

        left_points = int(lpt[i][0]), int(lpt[i][1])
        right_points = int(rpt[i][0]), int(rpt[i][1])
        cv2.circle(img1, left_points, 3, (255, 0, 0), 5)
        cv2.circle(img2, right_points, 3, (0, 0, 0), 5)
        fig1, axs1 = plt.subplots(1, 2, figsize=(20, 20))

        axs1[0].imshow(img1, cmap='gray')
        axs1[1].imshow(img2, cmap='gray')
        plt.tight_layout()
        savepath = 'transformation' + str(i) + 'png'
        plt.savefig(savepath)
        plt.show()


def test_segmentation():
    model_path = 'C:\\Users\\Samuel\\GoogleDrive\\Master\\complete\\CNN\\Trained_Models\\Pipe_symmetric.h5'
    model = load_model(model_path, custom_objects={'LogRegLoss': LogLoss()})

    x_test_path = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\test\\inp\\right'
    y_test_path = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\test\\targ'
    test_generator = cnn.create_generators(x_test_path, y_test_path, 1)
    (inp, target) = next(test_generator)

    p = cnn.get_prediction(model, inp)
    bi = binary_image(p[0], 0.2)
    di = dilate_image(bi, 5)
    k = blob_detector(di)
    im_with_keypoints = cv2.drawKeypoints(di, k, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    fig, axs = plt.subplots(1, 1, figsize=(20, 20))
    fig1, axs1 = plt.subplots(1, 2, figsize=(20, 20))
    fig2, axs2 = plt.subplots(1, 2, figsize=(20, 20))
    fig3, axs3 = plt.subplots(1, 2, figsize=(20, 20))

    # Training images
    axs1[0].imshow(inp[0])
    axs1[1].imshow(target[0].reshape(550, 400), cmap='gray')

    # Input and output
    axs2[0].imshow(inp[0])
    axs2[1].imshow(p[0], cmap='gray')

    # Binary image and dilation
    axs3[0].imshow(bi, cmap='gray')
    axs3[1].imshow(di, cmap='gray')

    # Blob detection
    axs.imshow(im_with_keypoints)

    plt.show()


def test_generation():
    x_test_path = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\test\\inp2'
    y_test_path = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\test\\targ'
    test_generator = cnn.create_generators(x_test_path, y_test_path, 1)
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    (inp, target) = next(test_generator)
    axs[0, 0].imshow(inp[0])

    (inp, target) = next(test_generator)
    axs[0, 1].imshow(inp[0])

    (inp, target) = next(test_generator)
    axs[1, 0].imshow(inp[0])
    (inp, target) = next(test_generator)
    axs[1, 1].imshow(inp[0])
    plt.tight_layout()
    plt.show()


def test_contact_points():
    model_path = 'C:\\Users\\Samuel\\GoogleDrive\\Master\\complete\\CNN\\Trained_Models\\Pipe_symmetric.h5'
    model = load_model(model_path, custom_objects={'LogRegLoss': LogLoss()})

    x_test_path = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\test\\inp\\right'
    y_test_path = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\test\\targ'
    test_generator = cnn.create_generators(x_test_path, y_test_path, 1)
    (inp, target) = next(test_generator)

    p = cnn.get_prediction(model, inp)
    bi = binary_image(p[0], 0.2)
    di = dilate_image(bi, 5)
    k = blob_detector(di)

    pt=(int(k[-1].pt[0]),int(k[-1].pt[1]))
    cp=find_contact_points(di,pt)
    print(cp)

#sq=create_square(5)
#print(sq)
test_contact_points()