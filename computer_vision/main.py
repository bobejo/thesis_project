import cnn
import image_manipulation as im
import grasp_finder as gf
import snap_pic as sp
import paths
import keras
import glob
import global_variables as gv
import image_registration as ir
import numpy as np
import cv2
from Loss import LogLoss
import matplotlib.pyplot as plt
# 843,664
def find_3D_point(model):
    """
    Finds the best grasping point and angle of it.
    :return: The 3D gripping point in robot base frame and the gripping angle.
    """
    "Take the images"
    l_img, r_img = sp.snap_pic('', paths.save_path)
    "Undistort and rectify images"

    "Crop the image"
    im.crop_images((paths.save_path + '\\*jpg'), gv.lcrop, gv.rcrop)
    crop_path = paths.crop_path + '\\cropped\\left*'
    crop_path_right = paths.crop_path + '\\cropped\\right*'
    cropped_left_path = glob.glob(crop_path)  # The path of left cropped image
    cropped_right_path = glob.glob(crop_path_right)
    cropped_left = cv2.imread(cropped_left_path[-1])
    cropped_right = cv2.imread(cropped_right_path[-1])
    "Load trained model and run prediction"


    cropped_left = cv2.resize(cropped_left, (400, 550))

    # model = keras.models.load_model(paths.model_path)
    model = keras.models.load_model(paths.simple_model_path, custom_objects={'LogRegLoss': LogLoss()})
    p = cnn.get_prediction(model, cropped_left.reshape(1,550,400,3))
    cv2.imwrite((paths.prediction + '\\prediction\\pred.jpg'), p[0])

    p = cv2.resize(p[0], (420, 580))

    "Convert to binary image and dilate"
    bi = im.binary_image(p, gv.threshold)
    di = im.dilate_image(bi, gv.dilation_size)

    "Find the 2d gripping point and angle for the left image"
    cont, gripping_point_left, gripping_angle, _ = gf.contour_detector(di)

    # cv2.waitKey(0)
    "The point is coordinate in the cropped image and need to be converted to the full image"
    # Add the cropping
    cv2.circle(cont, gripping_point_left, 3, (255, 0, 0), 3)
    gripping_point_left_full = (
        gripping_point_left[0] + gv.lcrop[1][0] - 1, gripping_point_left[1] + gv.lcrop[0][0] - 1)

    "Find the corresponding point of the right camera. Requires the matrix A and t from "
    "image_registration.find_image_transformation()"


    gripping_point_right_full = ir.affine_transformation(gv.A, gv.t, gripping_point_left_full)

    gripping_point_right_full = (int(round(gripping_point_right_full[0])), int(round(gripping_point_right_full[1])))
    #gripping_point_right_full=(843,664)

    l_img = cv2.imread(l_img)
    r_img = cv2.imread(r_img)
    cv2.circle(l_img, gripping_point_left_full, 3, (255, 0, 0), 3)
    cv2.circle(r_img, gripping_point_right_full, 3, (255, 0, 0), 3)


    "Finds the 3D point in camera frame, using triangulation, and convert to base frame."

    # Load camera matrices
    lcm = np.genfromtxt(paths.left_matrix_path)
    rcm = np.genfromtxt(paths.right_matrix_path)
    #lcm=gv.P1
    #rcm = gv.P2
    # Triangulate
    tri_camera = gf.triangulate_point(gripping_point_left_full, gripping_point_right_full, rcm, lcm)

    # Convert to base frame
    gripping_point_base = ir.base_transform(gv.T, tri_camera)

    print(tri_camera)
    print(gripping_point_base)
    print(gripping_angle)

    cv2.imshow('fullLeft', l_img)
    cv2.imshow('fullRight', r_img)
    cv2.imshow('cont', cont)
    cv2.waitKey(0)
    return gripping_point_base, gripping_angle


if __name__ == "__main__":
    # model = keras.models.load_model(paths.model_path)
    model = keras.models.load_model(paths.simple_model_path, custom_objects={'LogRegLoss': LogLoss()})
    for i in range(0,10):
        gripping_points, gripping_angle = find_3D_point(model)

